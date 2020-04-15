import numpy as np

from skimage.util import random_noise
from skimage.morphology import binary_erosion
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.cluster import normalized_mutual_info_score

from keras import backend as K


def get_vessel_iterator(directory, image_size=(512, 512), batch_size=8,
                        n_patches=16, patch_size=(64, 64), **aug_args):
    gen = ImageDataGenerator(**aug_args)
    it = gen.flow_from_directory(directory, batch_size=batch_size, target_size=image_size,
                                 color_mode='grayscale', class_mode=None)

    vess_it = VesselNetworkIterator(it, patch_size, n_patches)

    return vess_it


def mutual_information(im1, im2):
    # assumes images contain integer values in [0,255]
    X = im1.astype(float)
    Y = im2.astype(float)
    hist_2d, _, _ = np.histogram2d(X.ravel(), Y.ravel(), bins=255)
    pxy = hist_2d / float(np.sum(hist_2d))  # joint probability distribution

    px = np.sum(pxy, axis=1)  # marginal distribution for x over y
    py = np.sum(pxy, axis=0)  # marginal distribution for y over x

    Hx = - sum(px*np.log(px + (px == 0)))  # Entropy of X
    Hy = - sum(py*np.log(py + (py == 0)))  # Entropy of Y
    Hxy = np.sum(-(pxy*np.log(pxy+(pxy == 0))).ravel())  # Joint Entropy


    M = Hx + Hy - Hxy  # mutual information
    nmi = 2*(M/(Hx+Hy))  # normalized mutual information
    return nmi


class VesselNetworkIterator:

    def __init__(self, it, patch_size, n_patches):
        self.it = it
        self.n_patches = n_patches
        self.patch_size = patch_size

    def __iter__(self):
        return self

    def _get_patch(self, x, xi, yi):
        if K.image_data_format() == 'channels_first':
            return x[:, :, yi:yi+self.patch_size[0], xi:xi+self.patch_size[1]]
        elif K.image_data_format() == 'channels_last':
            return x[:, yi:yi+self.patch_size[0], xi:xi+self.patch_size[1], :]

    def _erode(self, x):
        """Return real and eroded patches."""
        if K.image_data_format() == 'channels_first':
            _, _, h, w = x.shape
            patches = np.zeros((self.n_patches, 1, self.patch_size[0], self.patch_size[1]))
            patches_eroded = np.zeros((self.n_patches, 1, self.patch_size[0], self.patch_size[1]))
        elif K.image_data_format() == 'channels_last':
            _, h, w, _ = x.shape
            patches = np.zeros((self.n_patches, self.patch_size[0], self.patch_size[1], 1))
            patches_eroded = np.zeros((self.n_patches, self.patch_size[0], self.patch_size[1], 1))

        x_eroded = x.copy()

        n_patches_i = np.random.randint(self.n_patches)

        # Random patches
        for i in range(n_patches_i):
            yi = np.random.randint(0, w - self.patch_size[0])
            xi = np.random.randint(0, w - self.patch_size[1])

            patches[i] = self._get_patch(x, xi, yi)[0]

            k = np.random.randint(0, 3) * 2 + 3
            selem = np.random.randint(0, 2, (k, k), dtype=np.uint8)

            if K.image_data_format() == 'channels_first':
                patches_eroded[i, 0] = binary_erosion(patches[i, 0], selem=selem)
                patches_eroded[i, 0] = random_noise(patches_eroded[i, 0], mode='s&p', salt_vs_pepper=np.random.uniform()*0.5)
                x_eroded[0, 0, yi:yi+self.patch_size[0], xi:xi+self.patch_size[1]] = patches_eroded[i, 0]
            elif K.image_data_format() == 'channels_last':
                patches_eroded[i, ..., 0] = binary_erosion(patches[i, ..., 0], selem=selem)
                patches_eroded[i, ..., 0] = random_noise(patches_eroded[i, ..., 0], mode='s&p', salt_vs_pepper=np.random.uniform()*0.5)
                x_eroded[0, yi:yi+self.patch_size[0], xi:xi+self.patch_size[1], 0] = patches_eroded[i, ..., 0]

        # Random erosions
        return x_eroded

    def _compute_metric(self, gt, eroded):
        y = np.zeros(gt.shape[0])

        for i, (gti, erodedi) in enumerate(zip(gt, eroded)):
            mi = mutual_information(gti, erodedi)
            y[i] = mi
        return y

    def __next__(self):
        x = next(self.it)
        x = x / 255.

        x_eroded = x.copy()
        metric = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            p = np.random.randint(10)
            if p > 0:
                x_eroded[i:i+1] = self._erode(x[i:i+1])
            else:
                x_eroded[i:i+1] = x[i:i+1]

        metric = self._compute_metric(x, x_eroded)

        return x_eroded, metric

    next = __next__
