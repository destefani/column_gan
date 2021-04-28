import os
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ArtChallengeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.files = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.files.iloc[index, 0])
        image = Image.open(img_path)
        target_path = os.path.join(self.root_dir, self.files.iloc[index, 1])
        target = Image.open(target_path)

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target