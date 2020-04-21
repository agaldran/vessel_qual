import sys
# from torchvision.models import resnet as resnet_imagenet
from . import resnet_in_channels_exposed as resnet_imagenet
# import resnet_in_planes_exposed as resnet_imagenet

import torch

def get_arch(model_name, in_channels=1, n_classes=1, pretrained=False):
    '''
    Classification options are 'resnet18', 'resnet_18_from_cifar', 'resnet50', 'resnet50_from_cifar',
    'resnext50', 'resnext101'; pretrained=False/True
    '''

    if model_name == 'resnet18':
        model = resnet_imagenet.resnet18(pretrained=pretrained, in_channels=in_channels)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
        # model.fc = torch.nn.Sequential(
        #                                torch.nn.Linear(num_ftrs, 512),
        #                                torch.nn.ReLU(),
        #                                torch.nn.Linear(512, n_classes)
        # )

    elif model_name == 'resnet50':
        model = resnet_imagenet.resnet50(pretrained=pretrained, in_channels=in_channels)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnext50':
        model = resnet_imagenet.resnext50_32x4d(pretrained=pretrained, in_channels=in_channels)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnext101':
        model = resnet_imagenet.resnext101_32x8d(pretrained=pretrained, in_channels=in_channels)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
    else: sys.exit('not a valid model_name, check models.get_model.py')

    return model
if __name__ == '__main__':
    import time
    batch_size = 2
    batch = torch.zeros([batch_size, 1, 512, 512], dtype=torch.float32)
    model = get_arch('resnet18', in_planes=1, n_classes=1)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('Forward pass (bs={:d}) when running in the cpu:'.format(batch_size))
    start_time = time.time()
    logits = model(batch)
    print("--- %s seconds ---" % (time.time() - start_time))

