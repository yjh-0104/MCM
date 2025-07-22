import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace
from network.mcm import MCM
from data import get_trainloader, get_testloader
from loss import RegistrationLoss
import pystrum.pynd.ndutils as nd

print(torch.__version__)
print(torch.backends.cudnn.enabled)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

output_dir = 'logs'
logger = TensorBoardLogger(name='mcm',save_dir = output_dir )

class CardiacMotionModel(pl.LightningModule):

    def __init__(self, hparams):
        """Initialize model, loss function, hyperparameters, and other variables."""
        super(CardiacMotionModel, self).__init__()

        self.params = hparams
        self.epochs = self.params.epochs
        self.clip_len = self.params.clip_len
        self.data_root = self.params.data_root
        self.dataset_type = self.params.dataset_type
        self.initlr = self.params.initlr
        self.train_batchsize = self.params.train_bs
        self.val_batchsize = self.params.val_bs
        self.weight_decay = self.params.weight_decay
        self.crop_size = self.params.crop_size
        self.num_workers = self.params.num_workers

        self.gt1s, self.gt2s, self.preds, self.fields = [], [], [], []

        self.train_losses, self.train_losses_smo, self.train_losses_sim_img = [], [], []

        self.model = MCM()
        self.criterion = RegistrationLoss(lambda_smooth=0.05)

        self.save_hyperparameters()

    def configure_optimizers(self):
        """Set up optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.initlr, betas=[0.9, 0.999])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)
        return [optimizer], [scheduler]

    def init_weight(self, ckpt_path=None):
        """Load weights from a checkpoint if provided."""
        if ckpt_path:
            checkpoint = torch.load(ckpt_path)
            state_dict = self.model.state_dict()
            checkpoint_model = {k: v for k, v in checkpoint.items() if k in state_dict.keys()}
            state_dict.update(checkpoint_model)
            self.model.load_state_dict(checkpoint_model, strict=False)

    def compute_dice(self, pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
        """Compute Dice score between predicted mask and target mask."""
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2. * intersection + epsilon) / (union + epsilon) * 100
        return dice.item()
    
    def jacobian_det(self, disp, mask):
        """Calculate percent of negative Jacobian determinants and mean absolute deviation from 1."""
        disp = disp.cpu().numpy().astype(np.float64)
        mask = mask.cpu().numpy().astype(np.float64)
        np.set_printoptions(threshold=np.inf)

        if disp.ndim == 4:
            disp = disp.transpose(1, 2, 3, 0)
        elif disp.ndim == 3:
            disp = disp.transpose(1, 2, 0)
        else:
            raise ValueError("Unsupported disp dimensions, must be 2D or 3D.")

        volshape = disp.shape[:-1]
        nb_dims = len(volshape)
        assert len(volshape) in (2, 3), 'flow must be 2D or 3D'

        grid_lst = nd.volsize2ndgrid(volshape)
        grid = np.stack(grid_lst, len(volshape))
        J = np.gradient(disp + grid)

        if nb_dims == 3:
            dx, dy, dz = J[0], J[1], J[2]
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
            jacobian_det = Jdet0 - Jdet1 + Jdet2
        else: 
            dfdx, dfdy = J[0], J[1]
            jacobian_det = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

        negative_jacobians = np.sum(jacobian_det < 0)
        total_elements = jacobian_det.size
        negative_percent = (negative_jacobians / total_elements) * 100

        J_in_mask = jacobian_det[mask > 0]
        abs_deviation = np.abs(J_in_mask - 1)
        mean_abs_deviation = np.mean(abs_deviation) if len(abs_deviation) > 0 else 0.0

        return negative_percent, mean_abs_deviation

    def training_step(self, batch, batch_idx):
        """Run a training step: forward, compute loss and log."""
        x1, x2 = batch
        x1, x2 = x1.cuda(), x2.cuda()

        deformed_x1, deformation_field = self.model(x1, x2)
        mid_frame_idx = x1.size(1) // 2

        loss, loss_smo, loss_sim_img = self.criterion.compute_loss(deformed_x1, x2, deformation_field, mid_frame_idx)

        self.log("train_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss_smo_step", loss_smo, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss_sim_img_step", loss_sim_img, prog_bar=True, on_step=True, on_epoch=False)

        self.train_losses.append(loss.detach())
        self.train_losses_smo.append(loss_smo.detach())
        self.train_losses_sim_img.append(loss_sim_img.detach())

        return {"loss": loss, "loss_smo": loss_smo, "loss_sim_img": loss_sim_img}

    def on_train_epoch_end(self):
        """After each epoch, log and print average training losses, then clear lists."""
        if self.train_losses:
            avg_loss = torch.stack(self.train_losses).mean()
            self.log("train_loss_epoch", avg_loss, prog_bar=True, sync_dist=True)

            avg_loss_smo = torch.stack(self.train_losses_smo).mean()
            self.log("train_loss_smo_epoch", avg_loss_smo, prog_bar=True, sync_dist=True)

            avg_loss_sim_img = torch.stack(self.train_losses_sim_img).mean()
            self.log("train_loss_sim_img_epoch", avg_loss_sim_img, prog_bar=True, sync_dist=True)

            print(f"Epoch {self.current_epoch} - Average Train Loss: {avg_loss:.6f}")

        self.train_losses.clear()
        self.train_losses_smo.clear()
        self.train_losses_sim_img.clear()

    def validation_step(self, batch, batch_idx):
        """Run a validation step: forward, compute loss, deformation, and optionally Dice."""
        self.model.eval()
        x1, x2, gt1, gt2 = batch
        x1, x2, gt1, gt2 = x1.cuda(), x2.cuda(), gt1.cuda(), gt2.cuda()
        bz, nf, nc, h, w = x1.shape
        gt1 = gt1.unsqueeze(2) # [1, 1, 1, 128, 128]

        deformed_x1, deformation_field = self.model(x1, x2) # [1, 5, 1, 128, 128], [1, 5, 2, 128, 128]
        mid_frame_idx = x1.size(1) // 2

        with torch.no_grad():
            val_loss, loss_smo, loss_sim_img = self.criterion.compute_loss(deformed_x1, x2, deformation_field, mid_frame_idx)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("val_loss_smo", loss_smo, prog_bar=True)
        self.log("val_loss_sim_img", loss_sim_img, prog_bar=True)

        deformation_field_mid = deformation_field[:, 0].unsqueeze(1)  # [1, 1, 2, 128, 128]
        
        # Only calculate dice if gt2 is not all zeros
        if torch.any(gt2):
            deformed_x1_gt = self.model.spatial_trans(gt1, deformation_field_mid, mode="nearest")  # [1, 1, 1, 128, 128]
            deformed_x1_gt = deformed_x1_gt.squeeze(0).squeeze(0)  # [1, 128, 128]
            deformation_field_mid_sq = deformation_field_mid.squeeze()  # [2, 128, 128]
            gt2 = gt2.squeeze(0)  # [1, 128, 128]
            gt1_sq = gt1.squeeze() # [128, 128]

            # Append deformed gt and gt1 for dice calculation
            self.fields.append(deformation_field_mid_sq) # [2, 128, 128]
            self.preds.append(deformed_x1_gt) # [1, 128, 128]
            self.gt2s.append(gt2) # [1, 128, 128]
            self.gt1s.append(gt1_sq) # [128, 128]
        else:
            deformed_x1_gt = None

        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        """Aggregate all validation results, compute and log metrics."""
        dice_scores = [self.compute_dice(pred, gt) for pred, gt in zip(self.preds, self.gt2s)]
        avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
        std_dice = np.std(dice_scores) if dice_scores else 0.0

        jacobian_neg_percents = []
        jacobian_minus_1 = []

        for field, gt1 in zip(self.fields, self.gt1s):
            neg_percent, mean_abs_deviation = self.jacobian_det(field, gt1)
            jacobian_neg_percents.append(neg_percent)
            jacobian_minus_1.append(mean_abs_deviation)

        avg_jacobian_neg_percent = sum(jacobian_neg_percents) / len(jacobian_neg_percents) if jacobian_neg_percents else 0.0
        std_jacobian_neg_percent = np.std(jacobian_neg_percents) if jacobian_neg_percents else 0.0
        avg_jacobian_minus_1 = sum(jacobian_minus_1) / len(jacobian_minus_1) if jacobian_minus_1 else 0.0
        std_jacobian_minus_1 = np.std(jacobian_minus_1) if jacobian_minus_1 else 0.0

        self.log("val_dice", avg_dice, prog_bar=True)
        self.log("val_dice_std", std_dice, prog_bar=True)
        self.log("val_jacobian_neg%", avg_jacobian_neg_percent, prog_bar=True)
        self.log("val_jacobian_neg%_std", std_jacobian_neg_percent, prog_bar=True)
        self.log("val_jacobian_minus_1", avg_jacobian_minus_1, prog_bar=True)
        self.log("val_jacobian_minus_1_std", std_jacobian_minus_1, prog_bar=True)

        print(f"Validation Epoch Dice: {avg_dice}")
        print(f"Validation Epoch Dice STD: {std_dice}")
        print(f"Validation Epoch Jacobian Negative Percentage: {avg_jacobian_neg_percent:.4f}")
        print(f"Validation Epoch Jacobian Negative Percentage STD: {std_jacobian_neg_percent:.4f}")
        print(f"Validation Epoch Jacobian Minus 1: {avg_jacobian_minus_1:.4f}")
        print(f"Validation Epoch Jacobian Minus 1 STD: {std_jacobian_minus_1:.4f}")

        self.preds.clear()
        self.gt1s.clear()
        self.gt2s.clear()
        self.fields.clear()

    def train_dataloader(self):
        return get_trainloader(
            self.data_root,
            batchsize=self.train_batchsize,
            trainsize=self.crop_size,
            clip_len=self.clip_len 
        )

    def val_dataloader(self):
        return get_testloader(
            self.data_root,
            batchsize=self.val_batchsize,
            trainsize=self.crop_size,
            data_type=self.dataset_type,
            clip_len=self.clip_len 
        )


def main():
    RESUME = False
    resume_checkpoint_path = None
    args = {
        'epochs': 200,
        'clip_len': 5, # number of frames
        'data_root': './dataset_example',
        'dataset_type': "Val",
        'train_bs': 8,
        'val_bs': 1,
        'initlr': 1e-4,
        'weight_decay': 0.01,
        'crop_size': 128,
        'num_workers': 8,
        'seed': 1234
    }

    torch.manual_seed(args['seed'])
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.benchmark = True

    hparams = Namespace(**args)
    model = CardiacMotionModel(hparams)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch', filename='epoch{epoch:03d}',
        auto_insert_metric_name=False, every_n_epochs=1,
        save_top_k=1, mode="max", save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        check_val_every_n_epoch=2, max_epochs=hparams.epochs, accelerator='gpu',
        devices=1, precision=32, logger=logger, strategy="auto",
        enable_progress_bar=True, log_every_n_steps=10,
        callbacks=[checkpoint_callback, lr_monitor_callback, early_stopping_callback]
    )

    trainer.fit(model, ckpt_path=resume_checkpoint_path)
    
if __name__ == '__main__':
    main()

