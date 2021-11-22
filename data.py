from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class SkinDataset(Dataset):
    def __init__(self, datapath, re_size, valid_test=False):
        self.image = [path_file for path_file in sorted(glob(datapath + '_Data/*.jpg'))]
        self.mask = [path_file for path_file in sorted(glob(datapath + '_Part1_GroundTruth/*.png'))]
        self.re_size = re_size
        self.valid_test = valid_test

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        x = cv2.imread(self.image[idx], cv2.IMREAD_COLOR)
        x = cv2.resize(x, (self.re_size[1], self.re_size[0]), interpolation=cv2.INTER_AREA)
        x = x / 255.
        x = torch.from_numpy(x).float().permute(2, 0, 1)

        y = cv2.imread(self.mask[idx], cv2.IMREAD_GRAYSCALE)
        origin_size = y.shape  # (height, width)
        y = cv2.resize(y, (self.re_size[1], self.re_size[0]), interpolation=cv2.INTER_AREA)
        y = y / 255.
        np.where(y > 0.5, 1, 0)
        y_h, y_w = y.shape
        y = torch.from_numpy(y).float().view(1, y_h, y_w)

        if self.valid_test:
            patient_id = self.mask[idx].split('/')[-1].split('_')[1]
            return {'image': x, 'mask': y, 'patient_id': patient_id, 'origin_h': origin_size[0],
                    'origin_w': origin_size[1]}
        else:
            return {'image': x, 'mask': y}


