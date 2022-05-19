import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from asset.data.base import ImagePaths, ImagePathsList


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, dir_seg=None, size_dataset=-1, max_ratio=0.85,
                 base_dir='data/landscape/imgs', data_type=''):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        paths = [os.path.join(base_dir, temp) for temp in paths]
        if size_dataset > 0:
            paths = paths[:size_dataset]
        #paths = paths[:120]  # to be deleted
        if data_type == '':
            self.data = ImagePaths(paths=paths, size=size, random_crop=True, dir_seg=dir_seg, max_ratio=max_ratio)
        elif data_type == 'list':
            self.data = ImagePathsList(paths=paths, size=size, random_crop=True, dir_seg=dir_seg, max_ratio=max_ratio)
        else:
            assert 0


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, dir_seg=None, size_dataset=-1, max_ratio=0.85,
                 base_dir='data/landscape/imgs', data_type=''):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        paths = [os.path.join(base_dir, temp) for temp in paths]
        if size_dataset > 0:
            paths = paths[:size_dataset]
        if data_type == '':  # single resolution, the basic one
            self.data = ImagePaths(paths=paths, size=size, random_crop=False, dir_seg=dir_seg, max_ratio=max_ratio)
        elif data_type == 'list':  # multiple resolutions
            self.data = ImagePathsList(paths=paths, size=size, random_crop=False, dir_seg=dir_seg, max_ratio=max_ratio)
        else:
            assert 0


