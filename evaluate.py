# Name: Oliver Fritsche (Team: Richard Zhao, Oliver Fritsche, Yunpei Gu)
# Class: CS 7180 Advanced Perception
# Date: 2025-09-21
# Purpose: Evaluation script and all related functions

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from PIL import Image
import os

def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal to Noise Ratio
    Args:
        img1, img2: tensors of shape (B, C, H, W) or (C, H, W) 
        max_val: maximum possible pixel value (1.0 for normalized images)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape, got {img1.shape} and {img2.shape}")
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, 
         data_range: float = 1.0) -> float:
    """
    Structural Similarity Index Measure
    Args:
        img1, img2: tensors of shape (B, C, H, W) or (C, H, W)
        window_size: size of the sliding window
        data_range: dynamic range of the images (1.0 for normalized images)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have same shape, got {img1.shape} and {img2.shape}")
    
    # Add batch dimension if needed
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Calculate means
    mu1 = img1.mean(dim=[2, 3], keepdim=True)
    mu2 = img2.mean(dim=[2, 3], keepdim=True)
    
    # Calculate variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2, 3], keepdim=True)

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_val = numerator / denominator
    return ssim_val.mean().item()


def calculate_metrics(enhanced_tensor, gt_tensor):
    """Calculate metrics comparing enhanced image to ground truth"""
    metrics = {}
    
    metrics['psnr'] = psnr(enhanced_tensor, gt_tensor)
    metrics['ssim'] = ssim(enhanced_tensor, gt_tensor)
    
    return metrics


