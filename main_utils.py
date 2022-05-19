import cv2
import glob
import os
import numpy as np
from PIL import Image


def write_cv2_img_jpeg(img, ofn):
    assert ofn.split('.')[-1] == 'jpeg'
    cv2.imwrite(ofn, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def txt2list(the_txt_fn):
    with open(the_txt_fn, 'r') as txt_obj:
        lines = txt_obj.readlines()
        lines = [haha.strip() for haha in lines]
    return lines


def list2txt(ofn, str_list):
    with open(ofn, 'a') as txt_obj:
        for small_str in str_list:
            txt_obj.write(small_str + '\n')


def show_img(cv2_array, ofn=None, title='image'):
    if ofn is None:
        cv2.imshow(title, cv2_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(ofn, cv2_array)


def folder2sslist(folder_name, file_type):
    ss_list = glob.glob(folder_name + '/*.' + file_type)
    ss_list = [os.path.split(haha)[1].split('.')[0] for haha in ss_list]
    return ss_list


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def file_exists(ifn):
    if os.path.isfile(ifn):
        return True
    else:
        return False


def fn2pr(ifn):
    return 1.0 - cv2.imread(ifn, cv2.IMREAD_GRAYSCALE) / 255.0


def pr2fn(pr, ofn=None):
    show_img(np.uint8(255.0 * (1.0 - pr)), ofn)


def show_segmentation(s, ofn):
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,None,:]  # h, w, 1, 182
    colorize = np.random.RandomState(1).randn(1,1,s.shape[-1],3)  # 1, 1, 182, 3
    colorize = colorize / colorize.sum(axis=2, keepdims=True)  # normalize
    s = s@colorize  # h, w, 1, 3
    s = s[...,0,:]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    s = Image.fromarray(s)
    s.save(ofn)
    
    
def show_output(s, ofn=None):
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    s = Image.fromarray(s)
    if ofn is not None:
        s.save(ofn)   
    return s


def laplacian_blend(source_img, target_img, mask, num_levels = 6):
    # assume mask is float32 [0,1]
    # generate Gaussian pyramid for A,B and mask
    GA = np.asarray(source_img)  # H, W, 3   in RGB
    GB = np.asarray(target_img)

    #mask = Image.fromarray(mask).resize((1024,1024), resample=Image.LANCZOS)
    GM = np.expand_dims(np.asarray(mask, dtype=np.float32), -1).repeat(3, axis=2)  # 256, 256, 3
    GM[GM < 0.5] = 0
    GM[GM != 0] = 1

    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)  # downsampling by 2
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)  # [0, 1] with anti-aliasing

        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks       len(gpA) = num_levels + 1  [0, ..., num_levels]
    lpA = [gpA[num_levels - 1]]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):  # [num_levels - 1, 1]
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
        
    ls_ = np.clip(ls_, 0, 255)
        
    return Image.fromarray(np.uint8(ls_))
        