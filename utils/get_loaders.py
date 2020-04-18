from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from . import paired_transforms_tv04 as p_tr
import torchvision.transforms as tr
import os.path as osp
import pandas as pd
from PIL import Image
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, selem
from skimage.metrics import structural_similarity as ssim, variation_of_information, mean_squared_error
from sklearn.metrics import f1_score
import random


def mutual_information(im1, im2):
    # assumes images contain integer values in [0,255]
    X = np.array(im1).astype(float)
    Y = np.array(im2).astype(float)
    hist_2d, _, _ = np.histogram2d(X.ravel(), Y.ravel(), bins=255)
    pxy = hist_2d / float(np.sum(hist_2d))  # joint probability distribution

    px = np.sum(pxy, axis=1)  # marginal distribution for x over y
    py = np.sum(pxy, axis=0)  # marginal distribution for y over x

    Hx = - sum(px * np.log(px + (px == 0)))  # Entropy of X
    Hy = - sum(py * np.log(py + (py == 0)))  # Entropy of Y
    Hxy = np.sum(-(pxy * np.log(pxy + (pxy == 0))).ravel())  # Joint Entropy

    M = Hx + Hy - Hxy  # mutual information
    nmi = 2 * (M / (Hx + Hy))  # normalized mutual information
    return nmi

class RegDataset(Dataset):
    def __init__(self, csv_path, p_manual=0.5, p_nothing=0.1, max_deg_patches=50, max_patch_size=(64, 64),
                 sim_method='mutual_info', transforms=None, tg_size=(512,512)):
        df = pd.read_csv(csv_path)
        self.p_manual = p_manual
        self.p_nothing = p_nothing
        self.max_deg_patches = max_deg_patches
        self.max_patch_size = max_patch_size
        self.vessels_list = df.vessel_paths
        self.mask_list = df.mask_paths
        self.sim_method = sim_method
        self.transforms = transforms
        self.rsz = p_tr.Resize(tg_size)

    def crop_to_fov(self, vessels, mask):
        minr, minc, maxr, maxc = regionprops(np.array(mask))[0].bbox
        vessels_crop = Image.fromarray(np.array(vessels)[minr:maxr, minc:maxc])
        return vessels_crop

    def compute_similarity(self, im, im_deg, sim_method='mutual_info'):
        im = np.array(im)
        im_deg = np.array(im_deg)
        if sim_method == 'mutual_info':
            return mutual_information(im, im_deg)
        elif sim_method == 'dice':
            return f1_score(im.astype(bool).ravel(), im_deg.astype(bool).ravel())
        elif sim_method == 'ssim':
            return ssim(im.astype(bool), im_deg.astype(bool))
        elif sim_method == 'var_info':
            under_seg, over_seg = variation_of_information(im.astype(bool), im_deg.astype(bool))
            return 1 - (under_seg + over_seg)
        elif sim_method == 'mse':
            return 1 - mean_squared_error(im.astype(bool).ravel(), im_deg.astype(bool).ravel())

    def erode_patch(self, patch):
        k = np.random.randint(0, 3) * 2 + 3  # 3, 5, 7
        return 255 * binary_erosion(patch, selem=selem.rectangle(k, k))

    def process_patch(self, im, xi, yi, patch_size):
        im[yi:yi + patch_size[0], xi:xi + patch_size[1]] = self.erode_patch(
            im[yi:yi + patch_size[0], xi:xi + patch_size[1]])
        return im

    def degrade_im(self, im, max_n_patches=100, max_patch_size=(64, 64)):
        im = np.array(im)
        h, w = im.shape
        n_patches = np.random.randint(max_n_patches)
        for i in range(n_patches):
            patch_size_h = np.random.randint(3, max_patch_size[0])  # min size of a patch is 3
            patch_size_w = np.random.randint(3, max_patch_size[1])

            xi = np.random.randint(0, h - patch_size_h)
            yi = np.random.randint(0, w - patch_size_w)

            im = self.process_patch(im, xi, yi, [patch_size_h, patch_size_w])
        return im

    def __getitem__(self, index):
        # load image and labels
        vessel_path = self.vessels_list[index]
        vessels_original = Image.open(vessel_path).convert('L')
        mask = Image.open(self.mask_list[index]).convert('L')
        vessels_original = self.crop_to_fov(vessels_original, mask)

        if random.random() > 1 - self.p_nothing:
            # we do not degrade, similarity=1, return now
            return p_tr.ToTensor()(self.rsz(vessels_original)), 1

        if random.random() > self.p_manual:
            # we return an artificial example
            epoch = random.choice([20, 40, 60])
            vessel_path = vessel_path.replace('manual/', 'predicted_epoch_' + str(epoch) + '/')
            vessels_pred = self.crop_to_fov(Image.open(vessel_path).convert('L'), mask)
            if self.transforms is not None:
                vessels_original, vessels_pred = self.transforms(vessels_original, vessels_pred)
            # thresholding only needed for non-binary images (predictions)
            threshold = 255 * (random.random() * 0.4 + 0.1)  # random threshold in [0.1,0.5]

            vessels_pred = np.array(vessels_pred) > threshold
            # we degrade, compute similarity later
            vessels_deg = self.degrade_im(vessels_pred, self.max_deg_patches, self.max_patch_size)
            # transform may introduce perturbations on binary ground-truth
            vessels_original = Image.fromarray(np.array(vessels_original) > 0.5)
        else:
            # we degrade, compute similarity later
            vessels_deg = self.degrade_im(vessels_original, self.max_deg_patches, self.max_patch_size)

        sim = self.compute_similarity(vessels_original, vessels_deg, sim_method=self.sim_method)
        return p_tr.ToTensor()(self.rsz(Image.fromarray(vessels_deg))), sim

    def __len__(self):
        return len(self.vessels_list)


class SegDataset(Dataset):
    def __init__(self, csv_path, transforms=None, label_values=None):
        self.csv_path = csv_path
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.gt_list = df.vessel_paths
        self.mask_list = df.mask_paths
        self.transforms = transforms
        self.label_values = label_values  # for use in label_encoding

    def label_encoding(self, gdt):
        gdt_gray = np.array(gdt.convert('L'))
        classes = np.arange(len(self.label_values))
        for i in classes:
            gdt_gray[gdt_gray == self.label_values[i]] = classes[i]
        return Image.fromarray(gdt_gray)

    def crop_to_fov(self, img, target, mask):
        minr, minc, maxr, maxc = regionprops(np.array(mask))[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        tg_crop = Image.fromarray(np.array(target)[minr:maxr, minc:maxc])
        mask_crop = Image.fromarray(np.array(mask)[minr:maxr, minc:maxc])
        return im_crop, tg_crop, mask_crop

    def __getitem__(self, index):
        # load image and labels
        img = Image.open(self.im_list[index])
        target = Image.open(self.gt_list[index])
        mask = Image.open(self.mask_list[index]).convert('L')

        img, target, mask = self.crop_to_fov(img, target, mask)
        target = self.label_encoding(target)

        target = np.array(self.label_encoding(target))
        target[np.array(mask) == 0] = 0
        target = Image.fromarray(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.im_list)

class TestDataset(Dataset):
    def __init__(self, csv_path, tg_size):
        df = pd.read_csv(csv_path)
        self.im_list = df.im_paths
        self.mask_list = df.mask_paths
        self.tg_size = tg_size

    def crop_to_fov(self, img, mask):
        mask = np.array(mask).astype(int)
        minr, minc, maxr, maxc = regionprops(mask)[0].bbox
        im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
        return im_crop, [minr, minc, maxr, maxc]

    def __getitem__(self, index):
        # load image and mask
        img = Image.open(self.im_list[index])
        mask = Image.open(self.mask_list[index]).convert('L')
        img, coords_crop = self.crop_to_fov(img, mask)
        original_sz = img.size[1], img.size[0]  # in numpy convention

        rsz = p_tr.Resize(self.tg_size)
        tnsr = p_tr.ToTensor()
        tr = p_tr.Compose([rsz, tnsr])
        img = tr(img)  # only transform image

        return img, np.array(mask).astype(bool), coords_crop, original_sz, self.im_list[index]

    def __len__(self):
        return len(self.im_list)

def get_seg_datasets(csv_path_train, csv_path_val, tg_size=(512, 512)):
    train_dataset = SegDataset(csv_path=csv_path_train, label_values=[0, 255])
    val_dataset = SegDataset(csv_path=csv_path_val, label_values=[0, 255])
    # transforms definition
    size = tg_size
    # required transforms
    resize = p_tr.Resize(size)
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=45, fill=(0,))
    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)
    # train_transforms = p_tr.Compose([resize,  scale_transl_rot, h_flip, v_flip, jitter, tensorizer])
    train_transforms = p_tr.Compose([resize, rotate, h_flip, v_flip, jitter, tensorizer])
    val_transforms = p_tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    return train_dataset, val_dataset


def get_reg_datasets(csv_path_train, csv_path_val, p_manual=0.5, p_nothing=0.1, max_deg_patches=50,
                     max_patch_size=(64, 64), sim_method='mutual_info', tg_size=(512, 512)):
    train_dataset = RegDataset(csv_path=csv_path_train, p_manual=p_manual, p_nothing=p_nothing,
                               max_deg_patches=max_deg_patches, max_patch_size=max_patch_size,
                               sim_method=sim_method, tg_size=tg_size)
    val_dataset = RegDataset(csv_path=csv_path_val, p_manual=p_manual, p_nothing=p_nothing,
                             max_deg_patches=max_deg_patches, max_patch_size=max_patch_size,
                             sim_method=sim_method, tg_size=tg_size)

    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=45, fill=(0,))
    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])

    train_transforms = p_tr.Compose([scale_transl_rot, h_flip, v_flip])
    # tensorize/resize inside dataset in this case
    train_dataset.transforms = train_transforms


    return train_dataset, val_dataset

def get_reg_loaders(csv_path_train, csv_path_val, batch_size=8, p_manual=0.5, p_nothing=0.1, max_deg_patches=50,
                     max_patch_size=(64, 64), sim_method='mutual_info', tg_size=(512, 512)):
    train_dataset, val_dataset = get_reg_datasets(csv_path_train, csv_path_val, p_manual=p_manual, p_nothing=p_nothing,
                                                  max_deg_patches=max_deg_patches, max_patch_size=max_patch_size,
                                                  sim_method=sim_method, tg_size=tg_size)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size * torch.cuda.device_count(),
    #                           num_workers=8, pin_memory=True, shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size * torch.cuda.device_count(),
    #                         num_workers=8, pin_memory=True, shuffle=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=8, pin_memory=True, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            num_workers=8, pin_memory=True, shuffle=False)
    return train_loader, val_loader

def get_seg_loaders(csv_path_train, csv_path_val, batch_size=8):
    train_dataset, val_dataset = get_seg_datasets(csv_path_train, csv_path_val)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size * torch.cuda.device_count(),
    #                           num_workers=8, pin_memory=True, shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size * torch.cuda.device_count(),
    #                         num_workers=8, pin_memory=True, shuffle=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=8, pin_memory=True, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            num_workers=8, pin_memory=True, shuffle=False)
    return train_loader, val_loader


def get_seg_test_dataset(csv_path='test.csv', tg_size=(512, 512)):
    # csv_path will only not be test.csv when we want to build training set predictions
    test_dataset = TestDataset(csv_path=csv_path, tg_size=tg_size)

    return test_dataset



