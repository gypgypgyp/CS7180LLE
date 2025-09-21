from torchvision.io import read_image
import os
from torch.utils.data import Dataset
import glob
import logging
import torch

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class SICEDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_files = glob.glob(os.path.join(self.img_dir, "*.jpg"))
        self.img_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.num_imgs = len(self.img_files)

        logger.info(f"Found {self.num_imgs} images in {self.img_dir}")

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        x = read_image(self.img_files[idx]).float() / 255.0
        return x
