import os

from PIL import Image

src_dir = 'test_1st_manual/imgs/'
dst_dir = 'test_1st_manual/imgs_png/'

for p in os.listdir(src_dir):
    src_path = os.path.join(src_dir, p)
    dst_path = os.path.join(dst_dir, p[:-3] + 'png')

    with open(src_path, 'rb') as f:
        img = Image.open(f)
        img.save(dst_path)
