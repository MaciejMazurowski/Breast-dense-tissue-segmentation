from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image 
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from torchvision import transforms


# # rotation and flip
# aug = A.Compose([
#     # A.Resize(512, 512),
#     # A.RandomCrop(width=256, height=256),
#     A.RandomRotate90(p=0.5),
#     A.HorizontalFlip(p=0.2),
#     A.VerticalFlip(p=0.2),
#     # A.RandomBrightnessContrast(p=0.2),
#     ToTensorV2()
# ])

# aug2 = A.Compose([
#     A.RandomSizedCrop(min_max_height=(480, 500), height=512, width=512, p=0.25),
#     A.RandomRotate90(p=0.5),
#     A.HorizontalFlip(p=0.2),
#     A.VerticalFlip(p=0.2),
#     A.RandomBrightnessContrast(brightness_limit=(0, 0.2), p=0.2),
#     # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3), #, brightness_by_max=True
#     ToTensorV2()
# ])

t = transforms.Compose([
    transforms.Resize(512), 
    transforms.ToTensor()
])
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale, img_transform, mask_transform, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(img):
        img = t(img)
        return img
    

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        return {
            'image': img,
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
