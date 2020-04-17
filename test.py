import torch
import time
from models.get_reg_model import get_arch

batch_size = 2
batch = torch.zeros([batch_size, 1, 512, 512], dtype=torch.float32)
model = get_arch('resnet18', in_channels=1, n_classes=1)
print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
print('Forward pass (bs={:d}) when running in the cpu:'.format(batch_size))
start_time = time.time()
logits = model(batch)
print("--- %s seconds ---" % (time.time() - start_time))
print(logits)
