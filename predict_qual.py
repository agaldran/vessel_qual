from skimage.measure import regionprops
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as tr

from models.get_reg_model import get_arch
from utils.reproducibility import set_seeds
from utils.model_saving_loading import load_model

import sys, os, argparse
from skimage import img_as_float
parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='better.gif', help='path to segmentation to evaluate')
parser.add_argument('--mask_path', type=str, default='12_mask.gif', help='path to corresponding mask')


def crop_to_fov(img, mask):
    mask = np.array(mask).astype(int)
    minr, minc, maxr, maxc = regionprops(mask)[0].bbox
    im_crop = Image.fromarray(np.array(img)[minr:maxr, minc:maxc])
    return im_crop

def prepare_single_image(im, mask):
    rsz = tr.Resize([512, 512])
    tnsr = tr.ToTensor()
    transf = tr.Compose([rsz, tnsr])

    im_crop = crop_to_fov(im, mask)
    return transf(im_crop).unsqueeze(0)

def get_score(im_path, mask_path, threshold=0.5):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # load image
    im = img_as_float(np.array(Image.open(im_path).convert('L')))  # this can be a grayscale or binary prediction
    im = im > threshold # this is binary for sure

    mask = img_as_float(Image.open(mask_path).convert('L')) # this should be a binary image
    mask = mask > 0.5 # this is binary for sure
    # load model
    model = get_arch('resnet18')
    load_checkpoint = 'experiments/best_mse/'
    model, _ = load_model(model, load_checkpoint, device=device, with_opt=False)
    model.eval()
    im_prep = prepare_single_image(im, mask)
    score = torch.sigmoid(model(im_prep.to(device))).item()
    return score

# if __name__ == '__main__':
#
#     args = parser.parse_args()
#     im_path = args.im_path
#     mask_path = args.im_path
#
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     seed_value = 0
#     set_seeds(seed_value, use_cuda)
#
#     # load image
#     THRESHOLD  = 100
#     im1 = Image.open(im_path).convert('L') # this is a grayscale prediction
#     im1 = np.array(im1)>THRESHOLD
#     mask = Image.open('01_mask.gif') # this is a binary image
#
#     # load model
#     model = get_arch('resnet18')
#     load_checkpoint = 'experiments/best_mse/'
#     model, _ = load_model(model, load_checkpoint, device=device, with_opt=False)
#     model.eval()
#
#     im_prep = prepare_single_image(im1, mask)
#     score1 = torch.sigmoid(model(im_prep.to(device))).item()
#     print('Segmentation score = {:.3f})'.format(score1))

