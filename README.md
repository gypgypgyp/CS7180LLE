# CS7180 LLE Adaptive Curve Estimation

Reimplements this paper https://arxiv.org/abs/2308.08197

**Teammates:** Richard Zhao(Training), Oliver Fritsche(Testing), Yunpei Gu (model)
**Course:** CS 7180 Advanced Perception  
**Operating System:** Linux 64, MacOS


## Project Overview

This project implements Stage1 of the Self-Reference Deep Adaptive Curve Estimation (Self-DACE) method for low-light image enhancement. The implementation is based on the Zero-DCE architecture and focuses on learning pixel-wise curve parameters for image enhancement.


## Data

Following the paper, we train using 

For training, we used SICE Part 1 dataset (2002 images), which is provided in  https://github.com/Developer-Zer0/ZeroDCE/tree/main/Dataset

For testing and evaluation, we used:
- Model Evaluation with Lol eval15 set: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
- Qualitative comparison used images from https://github.com/mahmoudnafifi/C5/tree/main/images and from author's repository https://github.com/John-Wendell/Self-DACE/blob/main/visualization/data


## Environment / Dependencies

To install dependencies
```
pip install -r requirements.txt
```

We used Python >3.10 and dependencies include:
- pytorch
- torchvision
- numpy
- matplotlib
- pillow
- jupyter / ipykernel
- tqdm

## How to run

### Option A — Notebook (for specific figures)

- Store your low-light images (recommend to use png format) in `data/random/low_num`
- (Optional) Store your high-light images in data/random/high_num
- Launch Jupyter and run `visualization_and_evaluation_special.ipynb`.

- Run the Config & model load cell (it expects epoch_118_model.pth at repo root).
- Run the Batch evaluation cell:

    Enhances all images in data/random/low_num.

    If a same-number GT exists in high_num, computes PSNR/SSIM.

    Saves triptychs to outputs/random/triptychs_num/.

- (Optional) Run the Iteration visualization cell to export “Original → Iter1..7 → Final”.

- The final Composite cell exports a single figure with Original | Self-DACE | GT rows. -->

### Option B — Notebook (for LOLdataset)
Open and run `visualization_and_evaluation.ipynb`

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

All hyperparameters are hardcoded as constants in that file.