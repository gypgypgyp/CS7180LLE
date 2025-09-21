# CS7180 LLE Adaptive Curve Estimation

Reimplements this paper https://arxiv.org/abs/2308.08197

**Teammates:** Richard Zhao(Training), Oliver Fritsche(Testing) , Yunpei Gu (model)
**Course:** CS 7180 Advanced Perception  

## Project Overview

This project implements Stage1 of the Self-Reference Deep Adaptive Curve Estimation (Self-DACE) method for low-light image enhancement. The implementation is based on the Zero-DCE architecture and focuses on learning pixel-wise curve parameters for image enhancement.


## Data

Following the paper, we train using the SICE Part 1 dataset (2002 images). 

We used the SICE Part 1 dataset, already resized to 512x512 size, provided in https://github.com/Developer-Zer0/ZeroDCE/tree/main/Dataset


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
