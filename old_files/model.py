"""Model methods."""
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import GlobalAveragePooling2D, Reshape, Input, Dense
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_input_vgg16
from keras.applications.vgg19 import VGG19, preprocess_input as preprocess_input_vgg19
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_input_v3
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet

from keras.layers import Conv2D, BatchNormalization

from keras import backend as K


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    return Conv2D(f, k, strides=s, padding=border_mode, **kwargs)


def BatchNorm(**kwargs):
    if K.image_data_format() == 'channels_first':
        axis = 1
    elif K.image_data_format() == 'channels_last':
        axis = -1

    return BatchNormalization(axis=axis, **kwargs)


def ConvBlock(i, nf, k=3, s=1, border_mode='same', drop_p=.0, norm=True, **kwargs):
    """A Conv-Pool-LeakyRelu-Batchnorm-Dropout block."""
    x = Convolution(nf, k=k, s=s, border_mode=border_mode, **kwargs)(i)
    x = LeakyReLU(0.02)(x)

    if norm:
        x = BatchNorm()(x)
    if drop_p > .0:
        x = Dropout(drop_p)(x)

    x = Convolution(nf, k=1, s=1, border_mode='same', **kwargs)(x)
    x = LeakyReLU(0.02)(x)

    if norm:
        x = BatchNorm()(x)
    if drop_p > .0:
        x = Dropout(drop_p)(x)

    return x


def mi_model(patch_size):

    input = Input(shape=patch_size + (1,))

    conv1 = ConvBlock(input, 32, s=2)
    conv2 = ConvBlock(conv1, 64, s=2)
    conv3 = ConvBlock(conv2, 128, s=2)
    conv4 = ConvBlock(conv3, 256, s=2)
    conv5 = ConvBlock(conv4, 512, s=2)
    conv6 = ConvBlock(conv5, 512)
    x = GlobalAveragePooling2D()(conv6)
    x = Dense(512)(x)
    x = LeakyReLU(0.02)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(input, out)
    opt = Adam(lr=2e-4)
    model.compile(optimizer=opt, loss='mae')

    return model


class MyModel:
    """Auxiliar class that helps with the creation of the model."""

    def __init__(self, architecture='inception', n_features=-1,
                 weak_sup=True, pretrain=True):
        """Create the model architecture."""
        self.n_features = n_features
        # Get the network architecture and corresponding preprocessing function
        m, self.preprocessing_fn = self._get_architecture(architecture)

        weights = 'imagenet' if pretrain else None
        self.base_model = m(weights=weights, include_top=False)

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        if K.image_data_format() == 'channels_first':
            target_shape = (-1, 1, 1)
        elif K.image_data_format() == 'channels_last':
            target_shape = (1, 1, -1)
        else:
            raise ValueError('{0} dim ordering is invalid or is not supported.'.format(K.image_data_format()))
        x = Reshape(target_shape)(x)

        if self.n_features > 0:
            x = ConvBlock(x, self.n_features, k=1, drop_p=.5)
        x = Convolution(1, k=1, activation='sigmoid')
        self.model = Model(self.base_model.input, x)

    def _get_architecture(self, architecture):
        arch_low = architecture.lower()
        if arch_low == 'inception':
            return InceptionV3, preprocess_input_v3
        if arch_low == 'vgg16':
            return VGG16, preprocess_input_vgg16
        if arch_low == 'vgg19':
            return VGG19, preprocess_input_vgg19
        if arch_low == 'resnet':
            return ResNet50, preprocess_input_resnet

        raise ValueError('Unknown architecture ({})'.format(architecture))

    def prepare_to_init(self, init_lr):
        """Set transfered layers untrainable and compile model."""
        for layer in self.base_model.layers:
            layer.trainable = False

        opt = Adam(lr=init_lr)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

        return self.model

    def prepare_to_finetune(self, fine_lr):
        """Set transfered layers trainable and compile model."""
        for layer in self.base_model.layers:
            layer.trainable = True

        opt = Adam(lr=fine_lr)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

        return self.model
