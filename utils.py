import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np

import os
import glob
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from math import sqrt

class UCMayo4(Dataset):
    """Ulcerative Colitis dataset grouped according to Endoscopic Mayo scoring system"""

    def __init__(self, root_dir, transform=None, val_split=0.2, random_seed=42, subset='train'):
        """
        root_dir (string): Path to the parent folder where class folders are located.
        transform (callable, optional): Optional transform to be applied on a sample.
        val_split (float): Proportion of the dataset used for validation (default is 0.2).
        random_seed (int): Random seed to ensure reproducibility of splits.
        subset (str): Which subset to return: 'train' or 'val'.
        """
        self.class_names = []
        self.samples = []
        self.transform = transform
        self.subset = subset

        # Set the random seed for reproducibility
        random.seed(random_seed)

        # Collect subfolders containing the classes
        subFolders = glob.glob(os.path.join(root_dir, "*"))
        subFolders.sort()
    
        # Extract class names and populate the samples list
        for folder in subFolders:
            className = os.path.basename(folder)
            self.class_names.append(className)

            image_paths = glob.glob(os.path.join(folder, "*"))
            for image_path in image_paths:
                self.samples.append((image_path, self.class_names.index(className)))

        # Shuffle the samples and split them into training and validation sets
        random.shuffle(self.samples)
        val_size = int(len(self.samples) * val_split)
        if subset == 'train':
            self.indices = self.samples[val_size:]
        else:
            self.indices = self.samples[:val_size]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, class_idx = self.indices[idx]
        image = Image.open(image_path).copy()
        if self.transform:
            image = self.transform(image)

        return image, class_idx

def get_dataset_mean_and_std(dataset_dir):
    trainingSet = UCMayo4(dataset_dir)
    R_total = 0
    G_total = 0
    B_total = 0

    total_count = 0
    for image, _ in trainingSet:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum(image[:, :, 0])
        G_total = G_total + np.sum(image[:, :, 1])
        B_total = B_total + np.sum(image[:, :, 2])

    R_mean = R_total / total_count
    G_mean = G_total / total_count
    B_mean = B_total / total_count

    R_total = 0
    G_total = 0
    B_total = 0

    total_count = 0
    for image, _ in trainingSet:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum((image[:, :, 0] - R_mean) ** 2)
        G_total = G_total + np.sum((image[:, :, 1] - G_mean) ** 2)
        B_total = B_total + np.sum((image[:, :, 2] - B_mean) ** 2)

    R_std = sqrt(R_total / total_count)
    G_std = sqrt(G_total / total_count)
    B_std = sqrt(B_total / total_count)

    return [R_mean / 255, G_mean / 255, B_mean / 255], [R_std / 255, G_std / 255, B_std / 255]