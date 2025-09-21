import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from model import light_net
from data import ImageDataset
from loss import TotalLoss
import logging
import time

BATCH_DIM = 0
CHANNEL_DIM = 1
HEIGHT_DIM = 2
WIDTH_DIM = 3

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# == Inputs ==
DATA_DIR = os.path.join("data", "sice_part1")

# Training associated hyperparameters
WEIGHT_DECAY = 1e-4
LR = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 200
GRAD_CLIP_NORM = 0.1
LR_SCHEDULER_GAMMA = 0.99

# Loss associated hyperparameters
W_LOCAL = 1000
W_GLOBAL = 1500
W_LUMINANCE = 5
W_ALPHA = 1000
W_BETA = 5000
Y = 0.8


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data
    train_ds = ImageDataset(img_dir=DATA_DIR, resize=(512, 512))
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Model
    model = light_net()
    model = model.to(device)

    # Optimizer
    opt = torch.optim.Adam(
        model.parameters(),
        weight_decay=WEIGHT_DECAY,
        lr=LR,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=1, gamma=LR_SCHEDULER_GAMMA
    )

    # Loss
    loss_fn = TotalLoss(
        w_local=W_LOCAL,
        w_global=W_GLOBAL,
        w_luminance=W_LUMINANCE,
        w_alpha=W_ALPHA,
        w_beta=W_BETA,
        y=Y,
    )

    # Logger
    writer = SummaryWriter()

    # Training
    start_time = time.perf_counter()
    epoch_idx = 0
    train_batch_idx = 0
    model.train()

    try:
        for epoch_idx in range(NUM_EPOCHS):
            logger.info(f"Epoch {epoch_idx}\n-------------------------------")

            train_dl_pbar = tqdm(train_dl, total=len(train_dl))
            train_dl_pbar.set_description("Train")

            for I_original in train_dl_pbar:  # (B, 3, H, W)
                I_original = I_original.to(device)

                # Extract estimated reflectance (color ratio) for original image
                per_pixel_intensity = torch.sum(I_original, dim=CHANNEL_DIM)
                E_original = I_original / (
                    per_pixel_intensity.unsqueeze(CHANNEL_DIM) + 1e-4
                )

                # Model Inference
                I_enhanced, alphas, betas = model(I_original)

                # Extract estimated reflectance (color ratio) for enhanced image
                per_pixel_intensity = torch.sum(I_enhanced, dim=CHANNEL_DIM)
                E_enhanced = I_enhanced / (
                    per_pixel_intensity.unsqueeze(CHANNEL_DIM) + 1e-4
                )

                # Loss
                loss = loss_fn(alphas, betas, I_enhanced, E_original, E_enhanced)

                # Update parameter
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), GRAD_CLIP_NORM)
                opt.step()

                # Log
                writer.add_scalar(
                    "Train Loss [every batch]", loss.item(), train_batch_idx
                )
                logger.info(f"Loss for batch {train_batch_idx}: {loss.item()}")
                train_batch_idx += 1

            # Update LR after each epoch
            scheduler.step()

        end_time = time.perf_counter()
        logger.info(f"Finished: Took {end_time - start_time}s")

    except KeyboardInterrupt:
        pass
    finally:
        # Save model either when interrupted or training done.
        output_model = f"epoch_{epoch_idx}_model.pth"
        logger.info(f"Saving Model before Quitting as {output_model}")
        torch.save(model.state_dict(), output_model)
