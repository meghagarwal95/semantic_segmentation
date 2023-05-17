
import os
import torch
import torchvision
from torch.utils.data import Dataset
from utils import rotateImg
import random, cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, hflip_prob=0., mean=[0.48897059, 0.46548275, 0.4294], std=[0.22861765, 0.22948039, 0.24054667], blur=False, rotate=False, val_split=0.3, size=(256,256)):
        self.horizontalFlipProb = hflip_prob
        self.mean = mean
        self.std = std
        self.rotate = rotate
        self.blur = blur
        self.size = size
        self.val_split = val_split
        # self.files = []
        self._get_files()

        #self._train_val_split()

    def _get_files(self):
        raise NotImplementedError

    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        if self.mode == "training":
            return len(self.files)

    # def _train_val_split(self):
    #     n = len(self.files)
    #     n_val = int(self.val_split*n)
    #     if self.val_split > 0:
    #         self.val_indices = random.sample(range(n), n_val)
    #     else:
    #         self.val_indices = []
    #     self.train_indices = list(set(range(n)) - set(self.val_indices))
    #     return

    def transform(self, image, label):
        (h, w) = image.shape[:2]

        if (h!=self.size[0]) and (w != self.size[1]):
            h, w = self.size
            image = cv2.resize(image, (w,h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w,h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

        if self.rotate:
            angle = random.randint(-10, 10)
            image = rotateImg(image, angle)
            label = rotateImg(label, angle, interp=cv2.INTER_NEAREST)

        # Random Horizontal Flip with a probability defined by user
        if self.horizontalFlipProb > 0:
            if random.random() < self.horizontalFlipProb:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blur (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            k_size = int(3.3 * sigma)
            k_size = k_size + 1 if k_size % 2 == 0 else k_size
            image = cv2.GaussianBlur(image, (k_size, k_size), sigmaX=sigma, sigmaY=sigma,
                                             borderType=cv2.BORDER_REFLECT_101)

        img_transform = transforms.Compose([
            # transforms.Resize(size,interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        label_transform = transforms.Compose([
            # transforms.Resize(size), #,interpolation=InterpolationMode.NEAREST_EXACT),
            transforms.ToTensor()
        ])

        img, label = img_transform(image), label_transform(label)
        return img, label

    def save_image(self, image, image_id):
        return

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)

        image, label = self.transform(image, label)

        return image, label




class BaseModel(torch.nn.Module):
    def __init__(self):
        pass

    def _load_model(self, num_classes):
        raise NotImplementedError