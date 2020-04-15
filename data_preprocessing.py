import os
import os.path as osp
from tqdm import tqdm
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from utils.image_preprocessing import correct_illumination


# handle paths
path_data_in= 'data/DRIVE/'
path_ims_in = osp.join(path_data_in, 'images/')
path_ims_out = osp.join(path_data_in, 'images_pre/')

path_csv_in = osp.join(path_data_in, 'train.csv')
path_csv_out = osp.join(path_data_in, 'train_pre.csv')

os.makedirs(path_ims_out, exist_ok=True)
df_all = pd.read_csv(path_csv_in)

im_list = df_all.im_paths.values
mask_list = df_all.mask_paths.values

for i in range(len(im_list)):
    im_n = im_list[i]
    mask_n = mask_list[i]
    print(im_n)
    img = io.imread(im_n)
    mask = io.imread(mask_n)

    img_pre = correct_illumination(img)
    img_pre[mask == 0] = 0
    im_n_out = im_n.replace('images', 'images_pre')
    io.imsave(im_n_out, img_pre)

im_list_new = [n.replace('images', 'images_pre') for n in im_list]

df_new = df_all.copy()
df_new.im_paths = im_list_new
df_new.to_csv(path_csv_out, index=False)
#################################################################################
# handle paths
path_data_in= 'data/DRIVE/'
path_ims_in = osp.join(path_data_in, 'images/')
path_ims_out = osp.join(path_data_in, 'images_pre/')

path_csv_in = osp.join(path_data_in, 'val.csv')
path_csv_out = osp.join(path_data_in, 'val_pre.csv')

os.makedirs(path_ims_out, exist_ok=True)
df_all = pd.read_csv(path_csv_in)

im_list = df_all.im_paths.values
mask_list = df_all.mask_paths.values

for i in range(len(im_list)):
    im_n = im_list[i]
    mask_n = mask_list[i]
    print(im_n)
    img = io.imread(im_n)
    mask = io.imread(mask_n)

    img_pre = correct_illumination(img)
    img_pre[mask == 0] = 0
    im_n_out = im_n.replace('images', 'images_pre')
    io.imsave(im_n_out, img_pre)

im_list_new = [n.replace('images', 'images_pre') for n in im_list]

df_new = df_all.copy()
df_new.im_paths = im_list_new
df_new.to_csv(path_csv_out, index=False)

