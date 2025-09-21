from torchvision.io import read_image
import os
from torch.utils.data import Dataset
import glob
import logging
import torch
from torchvision import transforms
from typing import Tuple

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, img_dir, resize: Tuple[int, int] = None):
        self.img_dir = img_dir
        self.img_files = glob.glob(os.path.join(self.img_dir, "*.jpg")) + glob.glob(
            os.path.join(self.img_dir, "*.png")
        )
        self.img_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.num_imgs = len(self.img_files)

        self.resize = None
        if resize:
            self.resize = transforms.Resize(resize)

        logger.info(f"Found {self.num_imgs} images in {self.img_dir}")

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        x = read_image(self.img_files[idx]).float() / 255.0
        if self.resize is not None:
            return self.resize(x)
        return x
