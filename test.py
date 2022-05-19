from main_utils import show_output, create_folder, show_img, write_cv2_img_jpeg
import os
from omegaconf import OmegaConf
import time
import yaml
from asset.models.cond_transformer import Net2NetTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from main_utils import laplacian_blend
import albumentations
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config_test_path', '-t', type=str, required=True, help='path to config file.')
parser.add_argument('--input_path', '-i', type=str, required=True, help='path to input image.')
parser.add_argument('--segmentation_path', '-s', type=str, required=True, help='path to seg map.')
parser.add_argument('--mask_path', '-m', type=str, required=True, help='path to mask.')
parser.add_argument('--category_name', '-c', type=str, required=True, help='new semantic category.')
parser.add_argument('--save_name', '-n', type=str, required=True, help='experiment name.')
parser.add_argument('--collect_dir', '-r', type=str, default='./results', help='directory to save results.')
args = parser.parse_args()

input_path = args.input_path
segmentation_path = args.segmentation_path
mask_path = args.mask_path
category_name = args.category_name
save_name = args.save_name
test_config = OmegaConf.load(args.config_test_path)
collect_dir = args.collect_dir

the_seed = test_config['the_seed']
config_path = test_config['config_path']
ckpt_path = test_config['ckpt_path']
guiding_ckpt_path = test_config['guiding_ckpt_path']
NUMBER_BATCHES = test_config['NUMBER_BATCHES']
NUMBER_SAMPLES = test_config['NUMBER_SAMPLES']
temperature = test_config['temperature']
top_k = test_config['top_k']
categories_dict = OmegaConf.to_container(test_config['categories_dict'])
number_categories = test_config['number_categories']

#----- 
torch.manual_seed(the_seed)
torch.cuda.manual_seed(the_seed)
torch.cuda.manual_seed_all(the_seed)  # if you are using multi-GPU.
np.random.seed(the_seed)  # Numpy module.
random.seed(the_seed)  # Python random module.
torch.manual_seed(the_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
  
config = OmegaConf.load(config_path)
config['model']['params']['ckpt_path'] = ckpt_path
config['model']['params']['guiding_ckpt_path'] = guiding_ckpt_path

print(yaml.dump(OmegaConf.to_container(config)))
model = Net2NetTransformer(**config.model.params)

model.cuda().eval()
torch.set_grad_enabled(False)

assert isinstance(top_k, int)
guide_scaler = albumentations.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA)

#-----------------------------
# --- load data
# ----------------------------

# ------ load mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = mask / 255.0
mask = mask.astype(np.float32)

iH = mask.shape[0]
iW = mask.shape[1]
assert iH == iW
assert iH in [512, 1024]  # work for 512 and 1024 square images
mask_final = mask.copy()
mask_final = np.clip(mask_final, 0, 1)
mask_uint8 = np.uint8(255.0 * mask_final)  # full mask
mask_tensor = torch.from_numpy(mask_final).to(model.device).unsqueeze(0).unsqueeze(0).to(memory_format=torch.contiguous_format)  # 1, 1, h, w, on cuda
resized_mask_tensor = F.interpolate(mask_tensor, size=(iH // 16, iW // 16))
latent_mask = resized_mask_tensor.squeeze().cpu().numpy()  # [0, 1]

# ------ load natural image
source = Image.open(input_path)
source = np.array(source).astype(np.uint8)
source_uint8 = source.copy()
source = (source / 127.5 - 1.0).astype(np.float32)
source = torch.tensor(source.transpose(2, 0, 1)[None]).to(dtype=torch.float32, device=model.device)  # 1, 3, h, w
z_code, z_indices = model.encode_to_z(source, mask_tensor=mask_tensor)  # VQGAN encoding

# ------ load segmentation
segmentation = Image.open(segmentation_path).convert('L')
segmentation = np.array(segmentation)  # uint8
assert segmentation.shape[0] == segmentation.shape[1]
if segmentation.shape[0] != source_uint8.shape[0]:  # resize the segmentation map to the same size as the natural image
    segmentation = cv2.resize(segmentation, (source_uint8.shape[1], source_uint8.shape[0]), interpolation=cv2.INTER_NEAREST)

new_cat_id = categories_dict[category_name]
if new_cat_id != -1:  # modify segmentation map
    segmentation = mask * new_cat_id + (1.0 - mask) * segmentation
    segmentation = np.uint8(segmentation)
segmentation_uint8 = segmentation.copy()
segmentation = np.eye(number_categories)[segmentation]  # convert to one-hot encoding
segmentation = torch.tensor(segmentation.transpose(2, 0, 1)[None]).to(dtype=torch.float32, device=model.device)  # 1, C, h, w

c_code, c_indices = model.encode_to_c(segmentation)  # segmentation encoding
print("c_code", c_code.shape, c_code.dtype)  # 1, 256, h//16, w//16
print("c_indices", c_indices.shape, c_indices.dtype)  # 1, h//16 * w//16
assert c_code.shape[2] * c_code.shape[3] == c_indices.shape[1]
z_indices_shape = c_indices.shape  # 1, 32x32
z_code_shape = c_code.shape  # 1, 256, 32, 32

# ------ get downsampled inputs
mask_guide = cv2.resize(mask_uint8, (256, 256), interpolation=cv2.INTER_NEAREST)
mask_guide = mask_guide / 255.0
mask_guide = mask_guide.astype(np.float32)
mask_guide_tensor = torch.from_numpy(mask_guide).to(model.device).unsqueeze(0).unsqueeze(0).to(memory_format=torch.contiguous_format)  # 1, 1, 256, 256, on cuda
resized_mask_guide_tensor = F.interpolate(mask_guide_tensor, size=(16, 16))
latent_mask_guide = resized_mask_guide_tensor.squeeze().cpu().numpy()  # [0, 1]

source_guide = guide_scaler(image=source_uint8)["image"]
source_guide = (source_guide / 127.5 - 1.0).astype(np.float32)
source_guide = torch.tensor(source_guide.transpose(2, 0, 1)[None]).to(dtype=torch.float32, device=model.device)  # 1, 3, h, w
z_code_guide, z_indices_guide = model.encode_to_z(source_guide, mask_tensor=mask_guide_tensor)  # VQGAN encoding

segmentation_guide = cv2.resize(segmentation_uint8, (256, 256), interpolation=cv2.INTER_NEAREST)
segmentation_guide = np.eye(number_categories)[segmentation_guide]
segmentation_guide = torch.tensor(segmentation_guide.transpose(2, 0, 1)[None]).to(dtype=torch.float32, device=model.device)  # 1, 182, h, w
c_code_guide, c_indices_guide = model.encode_to_c(segmentation_guide)  # 1, 256, 16, 16,       1, 256

#-----------------------------
# --- main part
# ----------------------------

# ------ create output dir
base_dir = create_folder(os.path.join(collect_dir, save_name))
write_cv2_img_jpeg(cv2.imread(input_path, cv2.IMREAD_COLOR), os.path.join(base_dir, 'input.jpeg'))
show_img(np.uint8(255.0 * mask_final), os.path.join(base_dir, 'mask.png'))

# ------ DO NOT CHANGE THE CODE BELOW
for bid in range(NUMBER_BATCHES):  # parallel sampling
    # ------ guiding synthesis
    guide_start_time = time.time()
    if model.is_SGA:
      fake_z_guide = model.autoregressive_sample_fast256(z_indices_guide, c_indices_guide,
                                                         c_code_guide.shape[2], c_code_guide.shape[3],
                                                         latent_mask_guide, batch_size=NUMBER_SAMPLES,
                                                         temperature=temperature, top_k=top_k)

      e_self_rand_attn, d_causal_rand_attn, d_cross_rand_attn = model.get_rough_attn_map(None, None, z_indices=fake_z_guide.reshape(-1, 256),
                                                                                                  c_indices=c_indices_guide.expand(NUMBER_SAMPLES,-1),
                                                                                                  resized_mask_tensor=resized_mask_guide_tensor.expand(NUMBER_SAMPLES,-1, -1,-1))
                                                                                                        
                                                                                                    
                                                                                                        
                                                                                                        
                                                                                                        
      print('guide synthesis %.2f seconds' % (time.time() - guide_start_time))
    # ------ high-resolution synthesis
    start_time = time.time()

    fake_z = model.autoregressive_sample_fast(z_indices, c_indices, c_code.shape[2], c_code.shape[3],
                                              latent_mask, batch_size=NUMBER_SAMPLES, temperature=temperature,
                                              top_k=top_k,
                                              e_self_rand_attn=e_self_rand_attn,
                                              d_causal_rand_attn=d_causal_rand_attn,
                                              d_cross_rand_attn=d_cross_rand_attn)

    fake_image = model.decode_to_img(fake_z, (fake_z.shape[0], z_code.shape[1], z_code.shape[2], z_code.shape[3]))

    for sid in range(fake_z.shape[0]):
        fake_img_pil = show_output(fake_image[sid:(sid + 1)])
        fake_img_pil_blended = laplacian_blend(fake_img_pil, Image.open(input_path), mask_final, num_levels=5)
        fake_img_np = np.array(fake_img_pil_blended)
        temp_ofn = os.path.join(base_dir, 's%d_s%02d.jpeg' % (the_seed, bid * NUMBER_SAMPLES + sid))
        write_cv2_img_jpeg(fake_img_np[:, :, ::-1], temp_ofn)

    print('target synthesis %.2f seconds' % (time.time() - start_time))
       
      



