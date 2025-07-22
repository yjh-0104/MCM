import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def find_global_min_max(folder_path):
    nii_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii.gz')])
    global_min, global_max = float('inf'), float('-inf')
    
    for file_path in nii_files:
        nii = nib.load(file_path)
        deformation_field = nii.get_fdata()
        global_min = min(global_min, np.min(deformation_field))
        global_max = max(global_max, np.max(deformation_field))
    
    return global_min, global_max


def deformation_field_to_rgb_custom(deformation_field, global_min, global_max):

    x_displacement = deformation_field[..., 0]
    y_displacement = deformation_field[..., 1]
    
    x_normalized = (x_displacement - global_min) / (global_max - global_min)
    y_normalized = (y_displacement - global_min) / (global_max - global_min)
    
    x_normalized = np.clip(x_normalized, 0, 1)
    y_normalized = np.clip(y_normalized, 0, 1)
    
    rgb = np.zeros((deformation_field.shape[0], deformation_field.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = x_normalized 
    rgb[..., 1] = y_normalized 
    rgb[..., 2] = 0.5 

    return rgb


def display_and_save_deformation_field(file_path, global_min, global_max):

    nii = nib.load(file_path)
    deformation_field = nii.get_fdata()
    
    if deformation_field.shape[-1] != 2:
        raise ValueError(f"deformation_field final dimetion should be 2, but {deformation_field.shape[-1]}")
    
    rgb_image = deformation_field_to_rgb_custom(deformation_field, global_min, global_max)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title(f"Deformation Field: {os.path.basename(file_path)}")
    plt.show()
    
    output_path = file_path.replace('.nii.gz', '_norm.png')
    plt.imsave(output_path, rgb_image)
    plt.close()


def display_and_save_all_deformation_fields(folder_path):
    global_min, global_max = find_global_min_max(folder_path)
    
    nii_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii.gz')])
    
    if not nii_files:
        print(f"file  {folder_path} no .nii.gz")
        return

    for file_path in nii_files:
        display_and_save_deformation_field(file_path, global_min, global_max)

#folder_path = "path/to/your/folder" 

#display_and_save_all_deformation_fields(folder_path)
