import os
import json
import albumentations
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from skimage import measure
from torch.utils.data import Dataset
Image.MAX_IMAGE_PIXELS = None


class CocoBase(Dataset):
    """needed for (image, caption, segmentation) pairs"""
    def __init__(self, size=None, dataroot="", datajson="", onehot_segmentation=False, use_stuffthing=False,
                 crop_size=None, force_no_crop=False, given_files=None, random_crop=False, seg_dir=None,
                 duplicate_num=1, size_dataset=-1):
        self.size = size
        if crop_size is None:
            self.crop_size = size
        else:
            self.crop_size = crop_size

        self.onehot = onehot_segmentation       # return segmentation as rgb or one hot
        self.stuffthing = use_stuffthing        # include thing in segmentation

        data_json = datajson
        with open(data_json) as json_file:
            self.json_data = json.load(json_file)
            self.img_id_to_captions = dict()
            self.img_id_to_filepath = dict()
            self.img_id_to_segmentation_filepath = dict()

        assert data_json.split("/")[-1] in ["captions_train2017.json",
                                            "captions_val2017.json"]
        if self.stuffthing:
            self.segmentation_prefix = seg_dir
        else:
            self.segmentation_prefix = (
                "data/coco/annotations/stuff_val2017_pixelmaps" if
                data_json.endswith("captions_val2017.json") else
                "data/coco/annotations/stuff_train2017_pixelmaps")

        imagedirs = self.json_data["images"]
        self.labels = {"image_ids": list()}
        for imgdir in tqdm(imagedirs, desc="ImgToPath"):
            self.img_id_to_filepath[imgdir["id"]] = os.path.join(dataroot, imgdir["file_name"])
            self.img_id_to_captions[imgdir["id"]] = list()
            pngfilename = imgdir["file_name"].replace("jpg", "png")
            self.img_id_to_segmentation_filepath[imgdir["id"]] = os.path.join(
                self.segmentation_prefix, pngfilename)
            if given_files is not None:
                if pngfilename in given_files:
                    self.labels["image_ids"].append(imgdir["id"])
            else:
                self.labels["image_ids"].append(imgdir["id"])
        if duplicate_num > 1:
            self.labels["image_ids"] = self.labels["image_ids"] * duplicate_num  # save checkpoints less frequently
        if size_dataset > 0:
            self.labels["image_ids"] = self.labels["image_ids"][:size_dataset]
            
        capdirs = self.json_data["annotations"]
        for capdir in tqdm(capdirs, desc="ImgToCaptions"):
            # there are in average 5 captions per image
            self.img_id_to_captions[capdir["image_id"]].append(np.array([capdir["caption"]]))

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size, interpolation=cv2.INTER_AREA)
        if not random_crop:
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
        else:
            self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
        if force_no_crop:
            self.rescaler = albumentations.Resize(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler],
                additional_targets={"segmentation": "image"})

    def __len__(self):
        return len(self.labels["image_ids"])

    def preprocess_image(self, image_path, segmentation_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)  # 427, 640, 3

        segmentation = Image.open(segmentation_path)
        if not self.onehot and not segmentation.mode == "RGB":
            segmentation = segmentation.convert("RGB")
        segmentation = np.array(segmentation).astype(np.uint8)  # 427, 640
        if self.onehot:
            assert self.stuffthing
            # stored in caffe format: unlabeled==255. stuff and thing from
            # 0-181. to be compatible with the labels in
            # https://github.com/nightrome/cocostuff/blob/master/labels.txt
            # we shift stuffthing one to the right and put unlabeled in zero
            # as long as segmentation is uint8 shifting to right handles the
            # latter too
            assert segmentation.dtype == np.uint8
            segmentation = segmentation + 1
        
        image = self.rescaler(image=image)["image"]
        segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        processed = self.cropper(image=image, mask=segmentation)
        image, segmentation = processed["image"], processed["mask"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        segmentation_uint8 = segmentation.copy()
        if self.onehot:
            assert segmentation.dtype == np.uint8
            # make it one hot
            n_labels = 183
            flatseg = np.ravel(segmentation)  # flattened array
            onehot = np.zeros((flatseg.size, n_labels), dtype=np.bool)
            onehot[np.arange(flatseg.size), flatseg] = True
            onehot = onehot.reshape(segmentation.shape + (n_labels,)).astype(np.float32)
            segmentation = onehot
        else:
            segmentation = (segmentation / 127.5 - 1.0).astype(np.float32)

        #----- generate masks
        mask = generate_mask_from_seg(segmentation_uint8, max_ratio=0.6)  # important function
        mask = (mask / 255.0).astype(np.float32)
        return image, segmentation, mask

    def __getitem__(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        seg_path = self.img_id_to_segmentation_filepath[self.labels["image_ids"][i]]
        image, segmentation, mask = self.preprocess_image(img_path, seg_path)
        captions = self.img_id_to_captions[self.labels["image_ids"][i]]
        # randomly draw one of all available captions per image
        caption = captions[np.random.randint(0, len(captions))]
        example = {"image": image,
                   "caption": [str(caption[0])],
                   "segmentation": segmentation,
                   "img_path": img_path,
                   "seg_path": seg_path,
                   "filename_": img_path.split(os.sep)[-1]
                    }
        if mask is not None:
            example['mask'] = mask
        return example


class CocoBaseList(Dataset):  # used for 512 and 1024 experiments
    """needed for (image, caption, segmentation) pairs"""
    def __init__(self, size=None, dataroot="", datajson="", onehot_segmentation=False, use_stuffthing=False,
                 crop_size=None, force_no_crop=False, given_files=None, random_crop=False, seg_dir=None,
                 duplicate_num=1, size_dataset=-1):
        self.size = size  # list
        if crop_size is None:
            self.crop_size = size
        else:
            self.crop_size = crop_size  # list

        self.onehot = onehot_segmentation       # return segmentation as rgb or one hot
        self.stuffthing = use_stuffthing        # include thing in segmentation

        data_json = datajson
        with open(data_json) as json_file:
            self.json_data = json.load(json_file)
            self.img_id_to_captions = dict()
            self.img_id_to_filepath = dict()
            self.img_id_to_segmentation_filepath = dict()

        assert data_json.split("/")[-1] in ["captions_train2017.json",
                                            "captions_val2017.json"]
        if self.stuffthing:
            self.segmentation_prefix = seg_dir
        else:
            self.segmentation_prefix = (
                "data/coco/annotations/stuff_val2017_pixelmaps" if
                data_json.endswith("captions_val2017.json") else
                "data/coco/annotations/stuff_train2017_pixelmaps")

        imagedirs = self.json_data["images"]
        self.labels = {"image_ids": list()}
        for imgdir in tqdm(imagedirs, desc="ImgToPath"):
            self.img_id_to_filepath[imgdir["id"]] = os.path.join(dataroot, imgdir["file_name"])
            self.img_id_to_captions[imgdir["id"]] = list()
            pngfilename = imgdir["file_name"].replace("jpg", "png")
            self.img_id_to_segmentation_filepath[imgdir["id"]] = os.path.join(
                self.segmentation_prefix, pngfilename)
            if given_files is not None:
                if pngfilename in given_files:
                    self.labels["image_ids"].append(imgdir["id"])
            else:
                self.labels["image_ids"].append(imgdir["id"])
        if duplicate_num > 1:
            self.labels["image_ids"] = self.labels["image_ids"] * duplicate_num  # save checkpoints less frequently
        if size_dataset > 0:
            self.labels["image_ids"] = self.labels["image_ids"][:size_dataset]
            
        capdirs = self.json_data["annotations"]
        for capdir in tqdm(capdirs, desc="ImgToCaptions"):
            # there are in average 5 captions per image
            self.img_id_to_captions[capdir["image_id"]].append(np.array([capdir["caption"]]))
        
        # ------ preprocessor
        self.scaler_list = []
        self.cropper_list = []
        for size_idx, size_hw in enumerate(self.size):
            self.scaler_list.append(albumentations.SmallestMaxSize(max_size=size_hw, interpolation=cv2.INTER_AREA))
            if size_idx == 0:
                if not random_crop:
                    self.cropper_list.append(albumentations.CenterCrop(height=self.crop_size[size_idx], width=self.crop_size[size_idx]))
                else:
                    self.cropper_list.append(albumentations.RandomCrop(height=self.crop_size[size_idx], width=self.crop_size[size_idx]))

    def __len__(self):
        return len(self.labels["image_ids"])

    def preprocess_image(self, image_path, segmentation_path):
        ans_dict = {}
        raw_dict = {}
        
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)  # 427, 640, 3

        segmentation = Image.open(segmentation_path)
        if not self.onehot and not segmentation.mode == "RGB":
            segmentation = segmentation.convert("RGB")
        segmentation = np.array(segmentation).astype(np.uint8)  # 427, 640
        if self.onehot:
            assert self.stuffthing
            # stored in caffe format: unlabeled==255. stuff and thing from
            # 0-181. to be compatible with the labels in
            # https://github.com/nightrome/cocostuff/blob/master/labels.txt
            # we shift stuffthing one to the right and put unlabeled in zero
            # as long as segmentation is uint8 shifting to right handles the
            # latter too
            assert segmentation.dtype == np.uint8
            segmentation = segmentation + 1
        
        # ----preprocessing
        image = self.scaler_list[0](image=image)["image"]  # the highest level
        segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        processed = self.cropper_list[0](image=image, mask=segmentation)
        raw_dict['raw_img_0'] = processed["image"]
        ans_dict["image_0"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        
        segmentation = processed["mask"]
        raw_dict['raw_seg_0'] = segmentation
        ans_dict["segmentation_0"] = np.eye(183)[segmentation]

        #--- lower res    no crop for now
        for size_idx, size_hw in enumerate(self.size):  # DO NOT use size
            if size_idx == 0:
                continue
            image_small = self.scaler_list[size_idx](image=raw_dict['raw_img_0'])["image"]
            raw_dict['raw_img_%d' % size_idx] = image_small
            ans_dict["image_%d" % size_idx] = (image_small/127.5 - 1.0).astype(np.float32)
            segmentation_small = cv2.resize(raw_dict['raw_seg_0'], (size_hw, size_hw), interpolation=cv2.INTER_NEAREST)
            raw_dict['raw_seg_%d' % size_idx] = segmentation_small
            ans_dict["segmentation_%d" % size_idx] = np.eye(183)[segmentation_small]
        


        #----- generate masks

        mask = generate_mask_from_seg(raw_dict['raw_seg_0'], max_ratio=0.6)  # important function
        ans_dict["mask_0"] = (mask / 255.0).astype(np.float32)
        #--- lower res    no crop for now
        for size_idx, size_hw in enumerate(self.size):
            if size_idx == 0:
                continue
            mask_small = cv2.resize(mask, (size_hw, size_hw), interpolation=cv2.INTER_NEAREST)
            ans_dict["mask_%d" % size_idx] = (mask_small / 255.0).astype(np.float32)
            
        ans_dict['raw_dict'] = raw_dict
        return ans_dict

    def __getitem__(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        seg_path = self.img_id_to_segmentation_filepath[self.labels["image_ids"][i]]
        example = self.preprocess_image(img_path, seg_path)

        example['filename_'] = img_path.split(os.sep)[-1]
        return example
    
    
class CocoImagesAndCaptionsTrain(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, size, onehot_segmentation=False, use_stuffthing=False, crop_size=None, force_no_crop=False):
        super().__init__(size=size,
                         dataroot="data/coco/train2017",
                         datajson="data/coco/annotations/captions_train2017.json",
                         onehot_segmentation=onehot_segmentation,
                         use_stuffthing=use_stuffthing, crop_size=crop_size, force_no_crop=force_no_crop)

    def get_split(self):
        return "train"


class CocoImagesAndCaptionsValidation(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, size, onehot_segmentation=False, use_stuffthing=False, crop_size=None, force_no_crop=False,
                 given_files=None):
        super().__init__(size=size,
                         dataroot="data/coco/val2017",
                         datajson="data/coco/annotations/captions_val2017.json",
                         onehot_segmentation=onehot_segmentation,
                         use_stuffthing=use_stuffthing, crop_size=crop_size, force_no_crop=force_no_crop,
                         given_files=given_files)

    def get_split(self):
        return "validation"


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
    if np.random.uniform() < 0.5:  # free-form mask
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
    else:  # bbox
        H = segmentation.shape[0]
        W = segmentation.shape[1]
        mask3 = np.zeros((H, W), dtype=np.uint8)
        x_set, y_set = np.where(mask2 == 1)  # pixel position
        x1, y1 = int(min(x_set) + 1), int(min(y_set) + 1)
        x2, y2 = int(max(x_set) + 1), int(max(y_set) + 1)
        hs, ws = x2 - x1, y2 - y1
        margin_w = int(max(round(ws / 100), 1))  # the larger object has larger margin
        margin_h = int(max(round(hs / 100), 1))
        x1, y1 = max(x1 - margin_h, 0), max(y1 - margin_w, 0)
        x2, y2 = min(x2 + margin_h, H), min(y2 + margin_w, W)
        mask3[x1:x2, y1:y2] = 255
    
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
