from skimage.measure import regionprops
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as tr

from models.get_reg_model import get_arch
from utils.reproducibility import set_seeds
from utils.model_saving_loading import load_model

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

if __name__ == '__main__':
    import time

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # load image examples
    THRESHOLD  = 100
    im1 = Image.open('worse.gif').convert('L') # this is a grayscale prediction
    im1 = np.array(im1)>THRESHOLD

    im2 = Image.open('better.gif').convert('L') # this is a grayscale prediction
    im2 = np.array(im2) > THRESHOLD

    im3 = Image.open('perfect.gif').convert('L') # this is a binary image

    mask = Image.open('01_mask.gif') # this is a binary image

    # load model
    model = get_arch('resnet18')
    load_checkpoint = 'experiments/best_mse/'
    model, _ = load_model(model, load_checkpoint, device=device, with_opt=False)

    # generate results
    t0 = time.time()
    im_prep = prepare_single_image(im1, mask)
    score1 = torch.sigmoid(model(im_prep.to(device))).item()
    print('Worst segmentation score = {:.3f} (inference time = {:.3f})'.format(score1, time.time()-t0))

    t0 = time.time()
    im_prep = prepare_single_image(im2, mask)
    score2 = torch.sigmoid(model(im_prep.to(device))).item()
    print('Better segmentation score = {:.3f} (inference time = {:.3f})'.format(score2, time.time()-t0))

    t0 = time.time()
    im_prep = prepare_single_image(im3, mask)
    score3 = torch.sigmoid(model(im_prep.to(device))).item()
    print('Perfect segmentation score = {:.3f} (inference time = {:.3f})'.format(score3, time.time() - t0))
