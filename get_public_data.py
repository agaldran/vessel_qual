import os
import os.path as osp
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision.transforms.functional import resize


print('downloading data')
call = '(mkdir data ' \
       '&& cd data ' \
       '&& curl https://codeload.github.com/sraashis/deepdyn/tar.gz/master | tar -xz --strip=2 deepdyn-master/data)'
os.system(call)
# call = '(wget https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip ' \
#        '&& unzip all.zip -d data/HRF ' \
#        '&& rm all.zip)'
# os.system(call)
# call = '(wget https://ignaciorlando.github.io/static/data/LES-AV.zip && unzip LES-AV.zip -d data/LES-AV ' \
#        '&& rm LES-AV.zip ' \
#        '&& rm -r data/LES-AV/__MACOSX)'
# os.system(call)
# call = '(mkdir data/LES_AV ' \
#        '&& mv data/LES-AV/LES-AV/images data/LES_AV/images ' \
#        '&& mv data/LES-AV/LES-AV/masks data/LES_AV/masks ' \
#        '&& mv data/LES-AV/LES-AV/vessel-segmentations data/LES_AV/manual' \
#        '&& rm -r data/LES-AV)'
# os.system(call)
# call ='wget http://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/DRHAGIS.zip ' \
#       '&& unzip DRHAGIS.zip -d data/DRHAGIS && rm DRHAGIS.zip && mv data/DRHAGIS/DRHAGIS data/DR_HAGIS ' \
#       '&& rm -r data/DRHAGIS && mv data/DR_HAGIS/Fundus_Images data/DR_HAGIS/images ' \
#       '&& mv data/DR_HAGIS/Mask_images data/DR_HAGIS/mask' \
#       '&& mv data/DR_HAGIS/Manual_Segmentations data/DR_HAGIS/manual' \
#       '&& rm data/DR_HAGIS/.DS_Store && rm data/DR_HAGIS/images/.DS_Store ' \
#       '&& rm data/DR_HAGIS/manual/.DS_Store && rm data/DR_HAGIS/mask/.DS_Store'
# os.system(call)
########################################################################################################################
# process drive data, generate CSVs
path_ims = 'data/DRIVE/manual'
path_masks = 'data/DRIVE/mask'

all_im_names = sorted(os.listdir(path_ims))
all_mask_names = sorted(os.listdir(path_masks))

# append paths
num_ims = len(all_im_names)
all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]

test_im_names = all_im_names[:num_ims//2]
train_im_names = all_im_names[num_ims//2:]

test_mask_names = all_mask_names[:num_ims//2]
train_mask_names = all_mask_names[num_ims//2:]

df_drive_all = pd.DataFrame({'vessel_paths': all_im_names,
                             'mask_paths': all_mask_names})

df_drive_train = pd.DataFrame({'vessel_paths': train_im_names,
                               'mask_paths': train_mask_names})

df_drive_test = pd.DataFrame({'vessel_paths': test_im_names,
                              'mask_paths': test_mask_names})

df_drive_train, df_drive_val = df_drive_train[:16], df_drive_train[16:]


df_drive_train.to_csv('data/DRIVE/train.csv', index=False)
df_drive_val.to_csv('data/DRIVE/val.csv', index=False)
df_drive_test.to_csv('data/DRIVE/test.csv', index=False)

shutil.rmtree('data/DRIVE/images')
print('DRIVE prepared')

# ########################################################################################################################
# src = 'missing_masks/chase-masks/'
# dst = 'data/CHASEDB/chase-masks/'
# # os.makedirs(dst, exist_ok=True)
# shutil.copytree(src, dst) #copytree creates the dir
#
# path_ims = 'data/CHASEDB/images'
# path_masks = 'data/CHASEDB/chase-masks'
# path_gts = 'data/CHASEDB/manual'
#
# all_im_names = sorted(os.listdir(path_ims))
# all_mask_names = sorted(os.listdir(path_masks))
# all_gt_names = sorted(os.listdir(path_gts))
#
# # append paths
# all_im_names = [osp.join(path_ims, n) for n in all_im_names]
# all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
# all_gt_names = [osp.join(path_gts, n) for n in all_gt_names if '1st' in n]
#
# num_ims = len(all_im_names)
# train_im_names = all_im_names[ :8]
# test_im_names  = all_im_names[8: ]
#
# train_mask_names = all_mask_names[ :8]
# test_mask_names  = all_mask_names[8: ]
#
# train_gt_names = all_gt_names[ :8]
# test_gt_names  = all_gt_names[8: ]
#
# df_chasedb_all = pd.DataFrame({'im_paths': all_im_names,
#                              'gt_paths': all_gt_names,
#                              'mask_paths': all_mask_names})
#
# df_chasedb_train = pd.DataFrame({'im_paths': train_im_names,
#                               'gt_paths': train_gt_names,
#                               'mask_paths': train_mask_names})
#
# df_chasedb_test = pd.DataFrame({'im_paths': test_im_names,
#                               'gt_paths': test_gt_names,
#                               'mask_paths': test_mask_names})
#
# num_ims = len(df_chasedb_train)
# tr_ims = int(0.8*num_ims)
# df_chasedb_train, df_chasedb_val = df_chasedb_train[:tr_ims], df_chasedb_train[tr_ims:]
#
# df_chasedb_train.to_csv('data/CHASEDB/train.csv', index=False)
# df_chasedb_val.to_csv('data/CHASEDB/val.csv', index=False)
# df_chasedb_test.to_csv('data/CHASEDB/test.csv', index=False)
# df_chasedb_all.to_csv('data/CHASEDB/test_all.csv', index=False)
# print('CHASE-DB prepared')
# ########################################################################################################################
# # process HRF data, generate CSVs
# path_ims = 'data/HRF/images'
# path_masks = 'data/HRF/mask'
# path_gts = 'data/HRF/manual1'
#
# path_ims_resized = 'data/HRF/images_resized'
# os.makedirs(path_ims_resized, exist_ok=True)
# path_masks_resized = 'data/HRF/mask_resized'
# os.makedirs(path_masks_resized, exist_ok=True)
# path_gts_resized = 'data/HRF/manual1_resized'
# os.makedirs(path_gts_resized, exist_ok=True)
#
# all_im_names = sorted(os.listdir(path_ims))
# all_mask_names = sorted(os.listdir(path_masks))
# all_gt_names = sorted(os.listdir(path_gts))
#
# # append paths
# num_ims = len(all_im_names)
# all_im_names = [osp.join(path_ims, n) for n in all_im_names]
# all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
# all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]
#
# df_hrf_all = pd.DataFrame({'im_paths': all_im_names,
#                             'gt_paths': all_gt_names,
#                             'mask_paths': all_mask_names})
#
# train_im_names = all_im_names[   :3*5]
# test_im_names =  all_im_names[3*5:   ]
#
# train_mask_names = all_mask_names[   :3*5]
# test_mask_names =  all_mask_names[3*5:   ]
#
# train_gt_names = all_gt_names[   :3*5]
# test_gt_names =  all_gt_names[3*5:   ]
#
#
# # use smaller images for trainining on HRF
# train_im_names_resized = [n.replace(path_ims, path_ims_resized) for n in train_im_names]
# train_mask_names_resized = [n.replace(path_masks, path_masks_resized) for n in train_mask_names]
# train_gt_names_resized = [n.replace(path_gts, path_gts_resized) for n in train_gt_names]
#
# df_hrf_train_fullRes = pd.DataFrame({'im_paths': train_im_names,
#                              'gt_paths': train_gt_names,
#                              'mask_paths': train_mask_names})
#
# df_hrf_train = pd.DataFrame({'im_paths': train_im_names_resized,
#                              'gt_paths': train_gt_names_resized,
#                              'mask_paths': train_mask_names_resized})
#
# df_hrf_test = pd.DataFrame({'im_paths': test_im_names,
#                               'gt_paths': test_gt_names,
#                                'mask_paths': test_mask_names})
#
# num_ims = len(df_hrf_train)
# tr_ims = int(0.8*num_ims)
# df_hrf_train_fullRes, df_hrf_val_fullRes = df_hrf_train_fullRes[:tr_ims], df_hrf_train_fullRes[tr_ims:]
# df_hrf_train, df_hrf_val = df_hrf_train[:tr_ims], df_hrf_train[tr_ims:]
#
# df_hrf_train.to_csv('data/HRF/train.csv', index=False)
# df_hrf_val.to_csv('data/HRF/val.csv', index=False)
# df_hrf_test.to_csv('data/HRF/test.csv', index=False)
# df_hrf_all.to_csv('data/HRF/test_all.csv', index=False)
#
# df_hrf_train_fullRes.to_csv('data/HRF/train_fullRes.csv', index=False)
# df_hrf_val_fullRes.to_csv('data/HRF/val_fullRes.csv', index=False)
#
# print('Resizing HRF images (**only** for training)\n')
# for i in tqdm(range(len(train_im_names))):
#     im_name = train_im_names[i]
#     im_name_out = train_im_names_resized[i]
#     im = Image.open(im_name)
#     im_res = resize(im, size=(im.size[1]//2, im.size[0]//2), interpolation=Image.BICUBIC)
#     im_res.save(im_name_out)
#
#     mask_name = train_mask_names[i]
#     mask_name_out = train_mask_names_resized[i]
#     mask = Image.open(mask_name)
#     mask_res = resize(mask, size=(mask.size[1]//2, mask.size[0]//2), interpolation=Image.NEAREST)
#     mask_res.save(mask_name_out)
#
#     gt_name = train_gt_names[i]
#     gt_name_out = train_gt_names_resized[i]
#     gt = Image.open(gt_name)
#     gt_res = resize(gt, size=(gt.size[1]//2, gt.size[0]//2), interpolation=Image.NEAREST)
#     gt_res.save(gt_name_out)
# print('HRF prepared')
# ########################################################################################################################
# src = 'missing_masks/stare-masks/'
# dst = 'data/STARE/stare-masks/'
# # os.makedirs(dst, exist_ok=True)
# shutil.copytree(src, dst) #copytree creates the dir
#
# path_ims = 'data/STARE/stare-images'
# path_masks = 'data/STARE/stare-masks'
# path_gts = 'data/STARE/labels-ah'
#
# all_im_names = sorted(os.listdir(path_ims))
# all_mask_names = sorted(os.listdir(path_masks))
# all_gt_names = sorted(os.listdir(path_gts))
#
# # append paths
# all_im_names = [osp.join(path_ims, n) for n in all_im_names]
# all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
# all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]
#
# df_stare_all = pd.DataFrame({'im_paths': all_im_names,
#                              'gt_paths': all_gt_names,
#                              'mask_paths': all_mask_names})
# df_stare_all.to_csv('data/STARE/test_all.csv', index=False)
# print('STARE prepared')
# ########################################################################################################################
# path_ims = 'data/AV-WIDE/images'
# path_masks = 'data/AV-WIDE/masks'
# os.makedirs(path_masks, exist_ok=True)
# path_gts = 'data/AV-WIDE/manual'
#
# test_im_names = sorted(os.listdir(path_ims))
# test_gt_names = sorted(os.listdir(path_gts))
#
# for n in test_im_names:
#     im = Image.open(osp.join(path_ims, n))
#     mask = 255*np.ones((im.size[1], im.size[0]), dtype=np.uint8)
#     Image.fromarray(mask).save(osp.join(path_masks, n))
#
# num_ims = len(all_im_names)
# test_mask_names = [osp.join(path_masks, n) for n in test_im_names]
# test_im_names = [osp.join(path_ims, n) for n in test_im_names]
# test_gt_names = [osp.join(path_gts, n) for n in test_gt_names]
#
# df_wide_test = pd.DataFrame({'im_paths': test_im_names,
#                               'gt_paths': test_gt_names,
#                               'mask_paths': test_mask_names})
#
# df_wide_test.to_csv('data/AV-WIDE/test_all.csv', index=False)
# print('AV-WIDE prepared')
# ########################################################################################################################
# path_ims = 'data/LES_AV/images'
# path_masks = 'data/LES_AV/masks'
# path_gts = 'data/LES_AV/manual'
#
# all_im_names = sorted(os.listdir(path_ims))
# all_mask_names = sorted(os.listdir(path_masks))
# all_gt_names = sorted(os.listdir(path_gts))
#
# all_im_names = [osp.join(path_ims, n) for n in all_im_names]
# all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
# all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]
#
# df_lesav_all = pd.DataFrame({'im_paths': all_im_names,
#                              'gt_paths': all_gt_names,
#                              'mask_paths': all_mask_names})
# df_lesav_all.to_csv('data/LES_AV/test_all.csv', index=False)
#
# print('LES-AV prepared')
# ########################################################################################################################
#
# path_ims = 'data/DR_HAGIS/images'
# path_masks = 'data/DR_HAGIS/mask'
# path_gts = 'data/DR_HAGIS/manual'
#
# all_im_names = sorted(os.listdir(path_ims), key=lambda s: s.split("_")[0] )
# all_mask_names = sorted(os.listdir(path_masks), key=lambda s: s.split("_")[0] )
# all_gt_names = sorted(os.listdir(path_gts), key=lambda s: s.split("_")[0] )
#
# all_im_names = [osp.join(path_ims, n) for n in all_im_names]
# all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
# all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]
#
# df_drhagis_all = pd.DataFrame({'im_paths': all_im_names,
#                              'gt_paths': all_gt_names,
#                              'mask_paths': all_mask_names})
# df_drhagis_all.to_csv('data/DR_HAGIS/test_all.csv', index=False)











# # create division to enable auc computation afterwards
# all_im_names = sorted(os.listdir(path_ims))
# all_mask_names = sorted(os.listdir(path_masks))
# all_gt_names = sorted(os.listdir(path_gts))
# num_ims = len(all_im_names)
#
# path_ims_1 = 'data/DR_HAGIS_1/images'
# path_masks_1 = 'data/DR_HAGIS_1/mask'
# path_gts_1 = 'data/DR_HAGIS_1/manual'
#
# os.makedirs(path_ims_1, exist_ok=True)
# os.makedirs(path_masks_1, exist_ok=True)
# os.makedirs(path_gts_1, exist_ok=True)
#
# path_ims_2 = 'data/DR_HAGIS_2/images'
# path_masks_2 = 'data/DR_HAGIS_2/mask'
# path_gts_2 = 'data/DR_HAGIS_2/manual'
#
# os.makedirs(path_ims_2, exist_ok=True)
# os.makedirs(path_masks_2, exist_ok=True)
# os.makedirs(path_gts_2, exist_ok=True)
#
# for i in range(num_ims // 2):
#     im_in = osp.join(path_ims, all_im_names[i])
#     im_out = osp.join(path_ims_1, all_im_names[i])
#     shutil.copyfile(im_in, im_out)
#
#     m_in = osp.join(path_masks, all_mask_names[i])
#     m_out = osp.join(path_masks_1, all_mask_names[i])
#     shutil.copyfile(m_in, m_out)
#
#     gt_in = osp.join(path_gts, all_gt_names[i])
#     gt_out = osp.join(path_gts_1, all_gt_names[i])
#     shutil.copyfile(gt_in, gt_out)
#
# all_im_names_ = sorted(os.listdir(path_ims_1))
# all_mask_names_ = sorted(os.listdir(path_masks_1))
# all_gt_names_ = sorted(os.listdir(path_gts_1))
#
# all_im_names_ = [osp.join(path_ims, n) for n in all_im_names_]
# all_mask_names_ = [osp.join(path_masks, n) for n in all_mask_names_]
# all_gt_names_ = [osp.join(path_gts, n) for n in all_gt_names_]
#
# df_dr_hagis_1 = pd.DataFrame({'im_paths': all_im_names_,
#                              'gt_paths': all_gt_names_,
#                              'mask_paths': all_mask_names_})
# df_dr_hagis_1.to_csv('data/DR_HAGIS_1/test_all.csv', index=False)
#
# for i in range(num_ims // 2, num_ims):
#     im_in = osp.join(path_ims, all_im_names[i])
#     im_out = osp.join(path_ims_2, all_im_names[i])
#     shutil.copyfile(im_in, im_out)
#
#     m_in = osp.join(path_masks, all_mask_names[i])
#     m_out = osp.join(path_masks_2, all_mask_names[i])
#     shutil.copyfile(m_in, m_out)
#
#     gt_in = osp.join(path_gts, all_gt_names[i])
#     gt_out = osp.join(path_gts_2, all_gt_names[i])
#     shutil.copyfile(gt_in, gt_out)
#
# all_im_names_ = sorted(os.listdir(path_ims_2))
# all_mask_names_ = sorted(os.listdir(path_masks_2))
# all_gt_names_ = sorted(os.listdir(path_gts_2))
#
# all_im_names_ = [osp.join(path_ims, n) for n in all_im_names_]
# all_mask_names_ = [osp.join(path_masks, n) for n in all_mask_names_]
# all_gt_names_ = [osp.join(path_gts, n) for n in all_gt_names_]
#
# df_dr_hagis_2 = pd.DataFrame({'im_paths': all_im_names,
#                              'gt_paths': all_gt_names,
#                              'mask_paths': all_mask_names})
# df_dr_hagis_2.to_csv('data/DR_HAGIS_2/test_all.csv', index=False)
#
# print('DR_HAGIS prepared')
########################################################################################################################
os.makedirs('experiments', exist_ok=True)
os.makedirs('results', exist_ok=True)

# remove junk
shutil.rmtree('data/VEVIO')
shutil.rmtree('data/DRIVE/splits')
shutil.rmtree('data/STARE/')
shutil.rmtree('data/CHASEDB/')
shutil.rmtree('data/AV-WIDE')
