# CS7180 LLE Adaptive Curve Estimation

Reimplements this paper https://arxiv.org/abs/2308.08197

**Teammates:** Richard Zhao(Training), Oliver Fritsche(Testing), Yunpei Gu (model)
**Course:** CS 7180 Advanced Perception  
**Operating System:** Linux 64, MacOS
Include a README file that includes your name, your teammates' names (if any), which OS you used, instructions for compiling and executing your program, and whether you want to use any of your time travel days.


## Project Overview

This project implements Stage1 of the Self-Reference Deep Adaptive Curve Estimation (Self-DACE) method for low-light image enhancement. The implementation is based on the Zero-DCE architecture and focuses on learning pixel-wise curve parameters for image enhancement.


## Data

Following the paper, we train using the SICE Part 1 dataset (2002 images). 

We used the SICE Part 1 dataset, already resized to 512x512 size, provided in https://github.com/Developer-Zer0/ZeroDCE/tree/main/Dataset

We also used dataset provided in this paper:
https://github.com/John-Wendell/Self-DACE/blob/main/visualization/data

To test the model in an environment with prominent colored light sources, we used this dataset:
https://github.com/mahmoudnafifi/C5/tree/main/images


## Environment / Dependencies

Python 3.10
PyTorch (match your local install)
(Optional) torchvision (only needed if you use data.ImageDataset)
numpy 1.26.4
matplotlib 3.8.4
pillow
jupyter / ipykernel

## How to run

### Option A — Notebook (for specific figures)

Store your low-light images(recommend to use png format) in data/random/low_num

(optional) Store your high-light images in data/random/high_num

Launch Jupyter and select the kernel

Open visualization_and_evaluation_special.ipynb.

Run the Config & model load cell (it expects epoch_118_model.pth at repo root).

Run the Batch evaluation cell:

Enhances all images in data/random/low_num.

If a same-number GT exists in high_num, computes PSNR/SSIM.

Saves triptychs to outputs/random/triptychs_num/.

(Optional) Run the Iteration visualization cell to export “Original → Iter1..7 → Final”.

The final Composite cell exports a single figure with Original | Self-DACE | GT rows.

### Option B — Notebook (for LOLdataset)

Open visualization_and_evaluation.ipynb

run each cell step by step


## Model

Model is in model.py

quick sanity test of model.py:

```bash
python model_test.py
```

Expected: prints shapes like:

```bash
enhanced: torch.Size([1, 3, 256, 256]) ...
alpha_stack: torch.Size([1, 21, 256, 256])
beta_stack : torch.Size([1, 21, 256, 256])
```

Used in training:
```bash
import torch
import model  # model.py

net = model.light_net().cuda().train()
# x0 in [0,1], shape Bx3xHxW
enhanced, alpha_stack, beta_stack = net(x0)
```

## Training

```
python train.py
```

Run tensorboard only if you're training locally `tensorboard --logdir=runs`
