# Name: Richard Zhao (Team: Richard Zhao, Oliver Fritsche, Yunpei Gu)
# Class: CS 7180 Advanced Perception
# Date: 2025-09-17
# Purpose: Define loss functions used in the paper.
# Notations:
# E = estimated reflectance map (essentially chromaticity/color ratio),
# I = image intensity normalized to unit range.

import torch
from torch import nn

BATCH_DIM = 0
CHANNEL_DIM = 1
HEIGHT_DIM = 2
WIDTH_DIM = 3


class LocalColorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, E_original: torch.Tensor, E_enhanced: torch.Tensor
    ) -> torch.Tensor:
        """
        Return loss that penalizes difference in reflectance map after enhancement, based on
        the retinex theory assumption that the reflectance map is
        invariant to illumination.

        Args:
            E_original (torch.Tensor of Shape(B, 3, H, W)): estimated reflectance map of the original image
            E_enhanced (torch.Tensor of Shape(B, 3, H, W)): estimated reflectance map of the enhanced image

        Returns:
            loss (scalar tensor)
        """

        per_sample_loss = torch.sum(
            (E_original - E_enhanced) ** 2, dim=CHANNEL_DIM
        )  # each sample is a pixel-wise loss (B, H, W)
        return per_sample_loss.mean()


class GlobalColorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, I_enhanced: torch.Tensor) -> torch.Tensor:
        """
        Returns loss that penalize deviation of output (enhanced) image from grayworld color
        constancy assumption.

        This loss is based on grayworld assumption: average illumination color should be gray.

        Args:
            I_enhanced (torch.Tensor of Shape(B, 3, H, W)): Enhanced image.

        Returns:
            loss (scalar tensor)
        """
        channel_intensity = torch.sum(I_enhanced, dim=(HEIGHT_DIM, WIDTH_DIM))  # (B, 3)
        total_intensity = torch.sum(channel_intensity, dim=CHANNEL_DIM) + 1e-4  # (B, )
        relative_channel_intensity = channel_intensity / total_intensity.unsqueeze(
            CHANNEL_DIM
        )  # A^c in paper, shape (B, 3)
        EXPECTED_REL_INTENSITY = 1 / 3

        per_sample_loss = torch.sum(
            (relative_channel_intensity - EXPECTED_REL_INTENSITY) ** 2, dim=CHANNEL_DIM
        )
        return per_sample_loss.mean()


class LuminanceLoss(nn.Module):
    def __init__(
        self,
        y=0.8,
    ):
        """
        Args:
            y (float, optional): Hyperparameter that tunes the adaptive intensity target.
            Defaults to 0.8.
        """
        super().__init__()
        self.y = y

    def forward(
        self, I_enhanced: torch.Tensor, E_original: torch.Tensor
    ) -> torch.Tensor:
        """
        Return loss that drives enhanced image intensity to
        an adaptive intensity target based on how far the original image's reflectance
        is from white chromaticity.

        Quoted from paper:
        "The closer the estimated reflectance of a pixel is to the center point,
        the higher brightness, while the expected luminance decreases
        when the estimation moves away from the center."

        Args:
            I_enhanced (torch.Tensor): enhanced image.
            E_original (torch.Tensor): estimated reflectance map of the original image.

        Returns:
            loss (scalar tensor)
        """
        per_pixel_intensity = torch.sum(I_enhanced, dim=CHANNEL_DIM) + 1e-4  # (B, H, W)

        WHITE_CHROMATICITY = 1 / 3
        per_pixel_dist_from_white_color = torch.sqrt(
            torch.sum((E_original - WHITE_CHROMATICITY) ** 2, dim=CHANNEL_DIM)
        )  # (B, H, W)

        per_pixel_intensity_target = (
            3 * self.y * (1 - per_pixel_dist_from_white_color)
        )  # H in paper of shape (B, H, W)

        per_sample_loss = (
            per_pixel_intensity - per_pixel_intensity_target
        ) ** 2  # each sample is a pixel-wise loss (B, H, W)

        return per_sample_loss.mean()


class CurveSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        Return loss that enforces smooth variation in feature `X`.

        In this model, it's used on "alpha" and "beta" per-pixel
        curve adjustment parameters to preserve spatial structure
        of the enhanced image.

        N = number of channels (3) * number of iterations of enhancement
        (e.g., 7 enhancement for 7-layer model)
        """
        B, N, H, W = X.shape
        grad_v = X[:, :, 1:, :] - X[:, :, :-1, :]  # (B, N, H-1, W)
        grad_h = X[:, :, :, 1:] - X[:, :, :, :-1]  # (B, N, H, W - 1)
        avg_grad_norm_squared = (
            1
            / (H * W)
            * torch.sum(grad_v.view(B, -1) ** 2 + grad_h.view(B, -1) ** 2, dim=-1)
        )  # (B, )

        return avg_grad_norm_squared.mean()


class TotalLoss(nn.Module):
    def __init__(
        self,
        w_local=1000,
        w_global=1500,
        w_luminance=5,
        w_alpha=1000,
        w_beta=5000,
        y=0.8,
    ):
        super().__init__()
        self.w_local = w_local
        self.w_global = w_global
        self.w_luminance = w_luminance
        self.w_alpha = w_alpha
        self.w_beta = w_beta

        self.LocalColorLoss = LocalColorLoss()
        self.GlobalColorLoss = GlobalColorLoss()
        self.LuminanceLoss = LuminanceLoss(y)
        self.CurveSmoothnessLoss = CurveSmoothnessLoss()

    def forward(self, alphas, betas, I_enhanced, E_original, E_enhanced):
        return (
            self.w_local * self.LocalColorLoss(E_original, E_enhanced)
            + self.w_global * self.GlobalColorLoss(I_enhanced)
            + self.w_luminance * self.LuminanceLoss(I_enhanced, E_original)
            + self.w_alpha * self.CurveSmoothnessLoss(alphas)
            + self.w_beta * self.CurveSmoothnessLoss(betas)
        )
