import bisect
import numpy as np
import albumentations
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
import os
from skimage import measure
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class ImagePaths(Dataset):  # used for VQGAN and guiding transformer
    def __init__(self, paths, size=None, random_crop=False, labels=None, dir_seg=None, max_ratio=0.85):
        self.n_labels = 182
        self.size = size
        self.random_crop = random_crop
        self.dir_seg = dir_seg
        self.max_ratio = max_ratio

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        self.rescaler = albumentations.SmallestMaxSize(max_size = self.size, interpolation=cv2.INTER_AREA)  # TODO  Use bicubic interpolation
        if not self.random_crop:
            self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
        else:
            self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        ans_dict = {}
        raw_dict = {}
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.rescaler(image=image)["image"]
        
        # load segmentation
        ss = os.path.split(image_path)[1].split('.')[0]
        seg_path = os.path.join(self.dir_seg, ss + '.png')
        segmentation = Image.open(seg_path)
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        if segmentation.shape[0] != image.shape[0]:  # resize the segmentation map to the same size as the natural image
            segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        assert self.cropper is not None
        processed = self.cropper(image=image, mask=segmentation)
        raw_dict['raw_img'] = processed["image"]
        raw_dict['raw_seg'] = processed["mask"]
        ans_dict["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation]
        ans_dict["segmentation"] = onehot
        
        mask = generate_mask_from_seg(segmentation, max_ratio=self.max_ratio)  # generate mask from segmentation
        ans_dict["mask"] = (mask / 255.0).astype(np.float32)
                  
        ans_dict['raw_dict'] = raw_dict
        return ans_dict

    def __getitem__(self, i):
        example = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class ImagePathsList(Dataset):  # more advanced dataset class for Sparse/Finetuning
    def __init__(self, paths, size=None, random_crop=False, labels=None, dir_seg=None, max_ratio=0.85):
        # size = [512, 256]      
        self.n_labels = 182
        self.size = size  # a list
        self.random_crop = random_crop
        self.dir_seg = dir_seg  # seg dir for highest resolution
        self.max_ratio = max_ratio

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)
        
        assert self.size is not None
        
        self.scaler_list = []
        self.cropper_list = []
        for size_idx, size_hw in enumerate(self.size):  # Two resolutions: high-resolution and low-resolution
            self.scaler_list.append(albumentations.SmallestMaxSize(max_size=size_hw, interpolation=cv2.INTER_AREA))
            if size_idx == 0:
                if not self.random_crop:
                    self.cropper_list.append(albumentations.CenterCrop(height=size_hw, width=size_hw))
                else:
                    self.cropper_list.append(albumentations.RandomCrop(height=size_hw, width=size_hw))
        
    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        ans_dict = {}
        raw_dict = {}
        
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)  # RGB
        image = self.scaler_list[0](image=image)["image"]  # high-res image
        
        # load segmentation
        ss = os.path.split(image_path)[1].split('.')[0]
        seg_path = os.path.join(self.dir_seg, ss + '.png')
        segmentation = Image.open(seg_path)
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        if segmentation.shape[0] != image.shape[0]:  # resize the segmentation map to the same size as the natural image
            segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)  # high-res seg
        processed = self.cropper_list[0](image=image, mask=segmentation)
        seg_for_mask = cv2.resize(processed['mask'], (512, 512), interpolation=cv2.INTER_NEAREST)  # used to generate masks
        
        raw_dict['raw_img_0'] = processed["image"]
        ans_dict["image_0"] = (processed["image"]/127.5 - 1.0).astype(np.float32)    
        segmentation = processed["mask"]    
        raw_dict['raw_seg_0'] = segmentation    
        ans_dict["segmentation_0"] = np.eye(self.n_labels)[segmentation]  # high-res one-hot seg    
            
        #--- lower res
        for size_idx, size_hw in enumerate(self.size):    
            if size_idx == 0:  # skip for the high-res
                continue
            image_small = self.scaler_list[size_idx](image=raw_dict['raw_img_0'])["image"]  # low-res image
            raw_dict['raw_img_%d' % size_idx] = image_small
            ans_dict["image_%d" % size_idx] = (image_small/127.5 - 1.0).astype(np.float32)
            segmentation_small = cv2.resize(seg_for_mask, (size_hw, size_hw), interpolation=cv2.INTER_NEAREST)
            raw_dict['raw_seg_%d' % size_idx] = segmentation_small        
            ans_dict["segmentation_%d" % size_idx] = np.eye(self.n_labels)[segmentation_small]    
                
        # generate mask
        mask = generate_mask_from_seg(seg_for_mask, max_ratio=self.max_ratio)
        mask = cv2.resize(mask, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)
        ans_dict["mask_0"] = (mask / 255.0).astype(np.float32)
        for size_idx, size_hw in enumerate(self.size):
            if size_idx == 0:
                continue
            mask_small = cv2.resize(mask, (size_hw, size_hw), interpolation=cv2.INTER_NEAREST)     
            ans_dict["mask_%d" % size_idx] = (mask_small / 255.0).astype(np.float32)

        ans_dict['raw_dict'] = raw_dict
        return ans_dict

    def __getitem__(self, i):
        example = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


def generate_mask_from_seg(segmentation, max_ratio=0.85, mask2=None):
    if mask2 is None:
        random_x = np.random.randint(0, high=segmentation.shape[0])
        random_y = np.random.randint(0, high=segmentation.shape[1])
        random_seg_id = segmentation[random_x, random_y]
    
        sampled_mask = (segmentation == random_seg_id) * 1  # np.int64, there might be multiple connected components
    
    
        labeled_mask = measure.label(sampled_mask, background=0)
        xnp, ynp = np.nonzero(labeled_mask)
        random_pixel_id = np.random.randint(0, high=xnp.shape[0])
        label_id = labeled_mask[xnp[random_pixel_id], ynp[random_pixel_id]]
        assert label_id > 0
        mask2 = (labeled_mask == label_id) * 1  # a connected component

    kernel = np.ones((5,5), np.uint8)
    assert segmentation.shape[0] in [256, 512, 1024]
    if segmentation.shape[0] == 256:
        mask3 = cv2.dilate(np.uint8(255.0 * mask2), kernel, iterations=2)
    elif segmentation.shape[0] == 512:  # adapt cv2.dilate according to resolution
        mask3 = cv2.dilate(np.uint8(255.0 * mask2), kernel, iterations=4)
    elif segmentation.shape[0] == 1024:  # adapt cv2.dilate according to resolution
        mask3 = cv2.dilate(np.uint8(255.0 * mask2), kernel, iterations=8)
    else:
        assert 0
    
    if np.sum(mask3 / 255.0) >= max_ratio * mask3.shape[0] * mask3.shape[1]:  # mask is too large
        h2 = mask3.shape[0] // 2
        w2 = mask3.shape[1] // 2
        random_int = np.random.randint(4)
        if random_int == 0:
            mask3[:h2, :] = 0
        elif random_int == 1:
            mask3[h2:, :] = 0
        elif random_int == 2:
            mask3[:, :w2] = 0
        else:
            mask3[:, w2:] = 0
    
    if np.sum(cv2.resize(mask3, (16, 16), interpolation=cv2.INTER_NEAREST) / 255.0) == 0.0:  # mask is too small
        h2 = mask3.shape[0] // 2
        w2 = mask3.shape[1] // 2
        mask3[:, :] = 0
        random_int = np.random.randint(4)
        if random_int == 0:
            mask3[:h2, :] = 255
        elif random_int == 1:
            mask3[h2:, :] = 255
        elif random_int == 2:
            mask3[:, :w2] = 255
        else:
            mask3[:, w2:] = 255
    return mask3