import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityLossImg:
    """Image similarity loss (MSE)."""
    def __init__(self):
        pass

    def compute_loss(self, y_true, y_pred):
        # Mean squared error between fixed and warped images
        similarity_loss = torch.mean((y_true - y_pred) ** 2)
        return similarity_loss

class SmoothnessLossDF:
    """Smoothness loss for deformation field."""
    def __init__(self, penalty='l2', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        # Compute spatial gradients along each dimension
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)
        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]
            
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def compute_loss(self, y_pred):
        # Compute L1 or L2 penalty on deformation gradients
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]
        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)
        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()

class RegistrationLoss:
    """Total registration loss: similarity + smoothness."""
    def __init__(self, lambda_smooth):
        self.similarity_loss_img_fn = SimilarityLossImg()
        self.smoothness_loss_DF_fn = SmoothnessLossDF()
        self.lambda_smooth = lambda_smooth

    def compute_loss(self, warped_moving_image, fixed_image, deformation_field, mid_frame_idx):

        # Use only the middle frame for loss calculation
        warped_moving_image_mid = warped_moving_image[:, 0]
        fixed_image_mid = fixed_image[:, mid_frame_idx]
        similarity_loss_img = self.similarity_loss_img_fn.compute_loss(warped_moving_image_mid, fixed_image_mid)
        deformation_field_mid = deformation_field[:, 0]
        smoothness_loss_DF = self.smoothness_loss_DF_fn.compute_loss(deformation_field_mid)

        # Combine losses
        total_loss = self.lambda_smooth * smoothness_loss_DF + similarity_loss_img
        return total_loss, smoothness_loss_DF, similarity_loss_img
