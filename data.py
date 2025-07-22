import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import nibabel as nib
from PIL import Image
from scipy.ndimage import zoom
import torchvision.transforms.functional as TF

def random_sync_augment(images1, images2, max_rotate=10):
    """Apply the same random flip and rotation to two lists of images."""
    do_hflip = random.random() > 0.5
    do_vflip = random.random() > 0.5
    angle = random.uniform(-max_rotate, max_rotate)
    aug_images1, aug_images2 = [], []
    for img1, img2 in zip(images1, images2):
        if do_hflip:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
        if do_vflip:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
        img1 = TF.rotate(img1, angle)
        img2 = TF.rotate(img2, angle)
        aug_images1.append(img1)
        aug_images2.append(img2)
    return aug_images1, aug_images2

def nii_to_image_frame(file_path):
    """Convert a nii file to a normalized PIL image for frame data."""
    target_spacing = (1.5, 1.5)
    crop_size = (128, 128)
    img = nib.load(file_path)
    img_data = img.get_fdata()
    current_spacing = img.header.get_zooms()[:2]
    resample_factors = [current_spacing[0] / target_spacing[0], current_spacing[1] / target_spacing[1]]
    resampled_img_data = zoom(img_data, resample_factors, order=3)
    target_h, target_w = crop_size
    center_h, center_w = resampled_img_data.shape[0] // 2, resampled_img_data.shape[1] // 2
    start_h = center_h - target_h // 2
    start_w = center_w - target_w // 2
    cropped = resampled_img_data[start_h:start_h + target_h, start_w:start_w + target_w]
    normalized = (cropped - cropped.min()) / (cropped.max() - cropped.min() + 1e-8)
    pil_image = Image.fromarray((normalized * 255).astype(np.uint8))
    return pil_image

def nii_to_image_gt(file_path):
    """Convert a nii file to a normalized PIL image for ground truth label."""
    target_spacing = (1.5, 1.5)
    crop_size = (128, 128)
    img = nib.load(file_path)
    img_data = img.get_fdata()
    current_spacing = img.header.get_zooms()[:2]
    resample_factors = [current_spacing[0] / target_spacing[0], current_spacing[1] / target_spacing[1]]
    resampled = zoom(img_data, resample_factors, order=0)
    target_h, target_w = crop_size
    center_h, center_w = resampled.shape[0] // 2, resampled.shape[1] // 2
    start_h = center_h - target_h // 2
    start_w = center_w - target_w // 2
    cropped = resampled[start_h:start_h + target_h, start_w:start_w + target_w]
    normalized = (cropped - cropped.min()) / (cropped.max() - cropped.min() + 1e-8)
    pil_image = Image.fromarray((normalized * 255).astype(np.uint8))
    return pil_image

class TrainDataset(data.Dataset):
    """Dataset for training. Loads frame pairs and applies synchronized augmentation."""
    def __init__(self, root, trainsize, clip_len):
        self.trainsize = trainsize
        image_root = os.path.join(root, "Train")
        vid_list = sorted(os.listdir(image_root))
        self.clip_len = clip_len
        self.first_frames, self.follow_frames = [], []
        for vid in vid_list:
            vid_path = os.path.join(image_root, vid, "Frame")
            frms = sorted([f for f in os.listdir(vid_path) if f.endswith('.nii.gz')])
            if len(frms) < 2:
                continue
            for idx in range(1, len(frms)):
                # Fixed frame sequence (always first frame, repeated)
                clip_first = [os.path.join(vid_path, frms[0])] * clip_len
                self.first_frames.append(clip_first)
                # Moving frame sequence (clip centered at current idx)
                clip_follow = []
                for ii in range(-clip_len // 2 + 1, clip_len // 2 + 1):
                    pick_idx = max(0, min(idx + ii, len(frms) - 1))
                    clip_follow.append(os.path.join(vid_path, frms[pick_idx]))
                self.follow_frames.append(clip_follow)
        self.img_transform = transforms.ToTensor()

    def __getitem__(self, index):
        first_images = [nii_to_image_frame(x) for x in self.first_frames[index]]
        follow_images = [nii_to_image_frame(x) for x in self.follow_frames[index]]
        first_images, follow_images = random_sync_augment(first_images, follow_images, max_rotate=10)
        first_images = torch.stack([self.img_transform(img) for img in first_images])
        follow_images = torch.stack([self.img_transform(img) for img in follow_images])
        return first_images, follow_images

    def __len__(self):
        return len(self.follow_frames)

class TestDataset(data.Dataset):
    """Dataset for testing/validation. Loads frame pairs and GT, handles ED/ES labeling."""
    def __init__(self, root, trainsize, data_type, clip_len):
        self.trainsize = trainsize
        self.data_type = data_type
        image_root = os.path.join(root, self.data_type)
        vid_list = sorted(os.listdir(image_root))
        self.clip_len = clip_len
        self.first_frames, self.follow_frames = [], []
        self.gts_first, self.gts_follow = [], []
        for vid in vid_list:
            vid_path = os.path.join(image_root, vid, "Frame")
            frms = sorted([f for f in os.listdir(vid_path) if f.endswith('.nii.gz')])
            if len(frms) < 2:
                continue
            gt_path = os.path.join(image_root, vid, "GT")
            ed_gt = os.path.join(gt_path, frms[0])
            es_gt_name = sorted([f for f in os.listdir(gt_path) if f.endswith('.nii.gz')])[-1]
            es_gt = os.path.join(gt_path, es_gt_name)
            es_frame_index = next((i for i, f in enumerate(frms) if f == es_gt_name), None)
            for idx in range(1, len(frms)):
                clip_first = [os.path.join(vid_path, frms[0])] * clip_len
                self.first_frames.append(clip_first)
                self.gts_first.append(ed_gt)
                clip_follow = []
                for ii in range(-clip_len // 2 + 1, clip_len // 2 + 1):
                    pick_idx = max(0, min(idx + ii, len(frms) - 1))
                    frame_path = os.path.join(vid_path, frms[pick_idx])
                    clip_follow.append(frame_path)
                self.follow_frames.append(clip_follow)
                # For ES labeling: only tag the center frame as ES, else None
                middle_frame_idx = clip_follow[len(clip_follow) // 2]
                self.gts_follow.append(es_gt if middle_frame_idx.endswith(frms[es_frame_index]) else None)
        self.img_transform = transforms.ToTensor()
        self.gt_transform = transforms.ToTensor()

    def __getitem__(self, index):
        first_images = [nii_to_image_frame(x) for x in self.first_frames[index]]
        ed_gt = nii_to_image_gt(self.gts_first[index]) if self.gts_first[index] else None
        follow_images = [nii_to_image_frame(x) for x in self.follow_frames[index]]
        es_gt = nii_to_image_gt(self.gts_follow[index]) if self.gts_follow[index] else None
        first_images = torch.stack([self.img_transform(img) for img in first_images])
        follow_images = torch.stack([self.img_transform(img) for img in follow_images])
        ed_gt = self.gt_transform(ed_gt) if ed_gt else torch.zeros(1, self.trainsize, self.trainsize)
        es_gt = self.gt_transform(es_gt) if es_gt else torch.zeros(1, self.trainsize, self.trainsize)
        return first_images, follow_images, ed_gt, es_gt

    def __len__(self):
        return len(self.follow_frames)

def get_trainloader(image_root, batchsize, trainsize, clip_len, shuffle=True, num_workers=12, pin_memory=True):
    """DataLoader for training."""
    dataset = TrainDataset(image_root, trainsize, clip_len)
    return data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

def get_testloader(image_root, batchsize, trainsize, data_type, clip_len, shuffle=False, num_workers=12, pin_memory=True):
    """DataLoader for testing/validation."""
    dataset = TestDataset(image_root, trainsize, data_type, clip_len)
    return data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
