#!/usr/bin/env python3
# Name: Yunpei Gu (Team: Richard Zhao, Oliver Fritsche, Yunpei Gu)
# Class: CS 7180 Advanced Perception
# Date: 2025-09-17
# Purpose: Stage-I Luminance-Net (7-layer vanilla CNN) for low-light enhancement.
# Notes:
#   - This file implements Stage-I only: a 7-layer CNN that regresses per-layer
#     AAC parameters (alpha_i, beta_i) and iteratively applies them to the same
#     input I0 to produce the enhanced image Ie.
#   - Outputs can be clamped to [0, 1] for safe image saving and loss computation. However, after training, the output is supposed to be in the [0, 1] range without clamping.

import torch
import torch.nn as nn
import numpy as np

# Constants
EPS = 1e-4 # Numerical epsilon to avoid division by zero

def reflection(image):
    """Compute per-pixel chromaticity (r,g,b) such that r+g+b=1.

        convert RGB to reflection [0,1]
        
        Args:
        image: Tensor [B, 3, H, W] in [0,1].

        Returns:
        Tensor [B, 3, H, W] with r=R/(R+G+B+eps), etc.
    """
    mr, mg, mb = torch.split(image, 1, dim=1)

    # Calculate red, green, blue reflectance
    denom = mr + mg + mb + EPS
    r = mr / denom
    g = mg / denom
    b = mb / denom

    return torch.cat([r, g, b], dim=1)

def luminance(s):
    """Compute a simple luminance proxy as the channel sum.

        convert RGB image to a single-channel luminance map

        Args:
        x: Tensor [B, 3, H, W].
        
        Returns:
        Tensor [B, 1, H, W] where L = R+G+B.
    """
    return (s[:, 0, :, :] + s[:, 1, :, :] + s[:, 2, :, :]).unsqueeze(1)


class light_net(nn.Module):
    """
    Stage-I Luminance-Net:
      - 7 convolutional layers, each: Conv(3x3, 32) + ReLU
        (the first layer is 3->32, the rest are 32->32)
      - From each layer, regress two 3-channel heads: 
        alpha_i (tanh) and beta_i (0.5+0.5*sigmoid)
      - Iteratively apply AAC to the same input I0 across i=1..7 to obtain the enhanced image Ie

    forward(x0) -> (enhanced, alpha_stack, beta_stack)
      enhanced:     B x 3 x H x W   (final Ie, clamped to [0,1])
      alpha_stack:  B x 21 x H x W  (concatenation of alpha_1..alpha_7)
      beta_stack:   B x 21 x H x W  (concatenation of beta_1..beta_7)
    """

    def __init__(self):
        super(light_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh() # outputs in [-1, 1]
        self.sigmoid = nn.Sigmoid() # outputs in [0, 1]
        number_f = 32 # number of features

        # 3×3 kernel, stride 1 (no downsampling), padding 1 (keeps H×W unchanged)
        # *_a: produces αᵢ (direction/strength) for R/G/B (3 channels).
        # *_b: produces βᵢ (targets/upper bounds) for R/G/B (3 channels).

        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv1_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv1_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv2_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv2_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv3_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv4_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv5 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv5_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv6 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv6_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv6_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self.e_conv7 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv7_a = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.e_conv7_b = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming init for conv."""
        for m in self.modules():
            # For every Conv2d, applies Kaiming (He) normal initialization to the weights
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            # For every BatchNorm2d, sets the scale (gamma) to 1 and the offset (beta) to 0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xo):
        """
        Forward pass of Stage-I AAC enhancement.
        Major sections:
          1) trunk feature extraction (7 layers)
          2) per-layer AAC heads (alpha/beta)
          3) iterative AAC update on the same input I0
          4) stack alpha/beta for losses; clamp output

        Args:
            xo: (B,3,H,W) low-light input in [0,1]
        Returns:
            (enhanced, alpha_stack, beta_stack)
        """

        x1 = self.relu(self.e_conv1(xo))
        x1_a = self.tanh(self.e_conv1_a(x1))
        # x1_b = self.sigmoid(self.e_conv1_a(x1)) * 0.5 + 0.5 # might be a bug in original repo
        x1_b = self.sigmoid(self.e_conv1_b(x1)) * 0.5 + 0.5

        x2 = self.relu(self.e_conv2(x1))
        x2_a = self.tanh(self.e_conv2_a(x2))
        x2_b = self.sigmoid(self.e_conv2_b(x2)) * 0.5 + 0.5

        x3 = self.relu(self.e_conv3(x2))
        x3_a = self.tanh(self.e_conv3_a(x3))
        x3_b = self.sigmoid(self.e_conv3_b(x3)) * 0.5 + 0.5

        x4 = self.relu(self.e_conv4(x3))
        x4_a = self.tanh(self.e_conv4_a(x4))
        x4_b = self.sigmoid(self.e_conv4_b(x4)) * 0.5 + 0.5

        x5 = self.relu(self.e_conv5(x4))
        x5_a = self.tanh(self.e_conv5_a(x5))
        x5_b = self.sigmoid(self.e_conv5_b(x5)) * 0.5 + 0.5

        x6 = self.relu(self.e_conv6(x5))
        x6_a = self.tanh(self.e_conv6_a(x6))
        x6_b = self.sigmoid(self.e_conv6_b(x6)) * 0.5 + 0.5

        x7 = self.relu(self.e_conv7(x6))  
        x7_a = self.tanh(self.e_conv7_a(x7))
        x7_b = self.sigmoid(self.e_conv7_b(x7)) * 0.5 + 0.5

        # xr = torch.cat([x1_a, x2_a, x3_a, x4_a, x5_a, x6_a, x7_a], dim=1)  # , x6_a, x7_a
        # xr1 = torch.cat([x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b], dim=1)  # , x6_b, x7_b

        xr = [x1_a, x2_a, x3_a, x4_a, x5_a, x6_a, x7_a]
        xr1 = [x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b]

        # for i in np.arange(7):
        #     # xo = xo + xr[:, 3 * i:3 * i + 3, :, :] * torch.maximum(xo * (xr1[:, 3 * i:3 * i + 3, :, :] - xo) * (1 / xr1[:, 3 * i:3 * i + 3, :, :]),0*xo)
        #     xo = xo + xr[:, 3 * i:3 * i + 3, :, :] * 1 / (
        #                 1 + torch.exp(-10 * (-xo + xr1[:, 3 * i:3 * i + 3, :, :] - 0.1))) * xo * (
        #                      xr1[:, 3 * i:3 * i + 3, :, :] - xo) * (1 / xr1[:, 3 * i:3 * i + 3, :, :])
        
        for r, r1 in zip(xr, xr1):
            xo = xo + r / (1 + torch.exp(-10 * (-xo + r1 - 0.1))) * xo * (r1 - xo) / r1

        # xo = xo.clamp(0, 1) # after training, the output is supposed to be in the [0, 1] range without clamping.

        xr = torch.cat([x1_a, x2_a, x3_a, x4_a, x5_a, x6_a, x7_a], dim=1)  
        xr1 = torch.cat([x1_b, x2_b, x3_b, x4_b, x5_b, x6_b, x7_b], dim=1)  

        return xo, xr, xr1


