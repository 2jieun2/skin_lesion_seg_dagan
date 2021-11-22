from glob import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage import rotate


def flip_by_axis(x, y, axis):
    x = np.flip(x, axis=axis)
    y = np.flip(y, axis=axis)
    return x, y


def rotation(x, y, angle):
    x = rotate(x, angle, reshape=False)
    y = rotate(y, angle, reshape=False)
    return x, y


train_x_path = '/home/ubuntu/jelee/dataset/skin_ISIC/ISIC-2017_Training_Data'
train_y_path = '/home/ubuntu/jelee/dataset/skin_ISIC/ISIC-2017_Training_Part1_GroundTruth'

x_path_list = [path_file for path_file in sorted(glob(train_x_path + '/ISIC_???????.jpg'))]
y_path_list = [path_file for path_file in sorted(glob(train_y_path + '/ISIC_???????_segmentation.png'))]


for idx, y_path in enumerate(y_path_list):
    print(f'-----{idx} / {len(y_path_list)}-----')
    x = cv2.imread(x_path_list[idx], cv2.IMREAD_COLOR)
    # y = cv2.imread(y_path_list[idx], cv2.IMREAD_GRAYSCALE)
    y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
    patient_id = y_path.split('/')[-1].split('_')[1]

    print('-----flip 0 (by x-axis)-----')
    flip_0 = flip_by_axis(x, y, 0)
    cv2.imwrite(train_x_path + '/Augment_filp0_ISIC_' + patient_id + '.jpg', flip_0[0])
    cv2.imwrite(train_y_path + '/Augment_filp0_ISIC_' + patient_id + '_segmentation.png', flip_0[1])

    print('-----flip 1 (by y-axis)-----')
    flip_1 = flip_by_axis(x, y, 1)
    cv2.imwrite(train_x_path + '/Augment_filp1_ISIC_' + patient_id + '.jpg', flip_1[0])
    cv2.imwrite(train_y_path + '/Augment_filp1_ISIC_' + patient_id + '_segmentation.png', flip_1[1])

    print('-----rot 90-----')
    rot_90 = rotation(x, y, 90)
    cv2.imwrite(train_x_path + '/Augment_rot90_ISIC_' + patient_id + '.jpg', rot_90[0])
    cv2.imwrite(train_y_path + '/Augment_rot90_ISIC_' + patient_id + '_segmentation.png', rot_90[1])

    print('-----rot 180-----')
    rot_180 = rotation(x, y, 180)
    cv2.imwrite(train_x_path + '/Augment_rot180_ISIC_' + patient_id + '.jpg', rot_180[0])
    cv2.imwrite(train_y_path + '/Augment_rot180_ISIC_' + patient_id + '_segmentation.png', rot_180[1])

    print('-----rot 270-----')
    rot_270 = rotation(x, y, 270)
    cv2.imwrite(train_x_path + '/Augment_rot270_ISIC_' + patient_id + '.jpg', rot_270[0])
    cv2.imwrite(train_y_path + '/Augment_rot270_ISIC_' + patient_id + '_segmentation.png', rot_270[1])


