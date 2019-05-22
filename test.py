import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as tr
import matplotlib.pyplot as plt

from mmcv import imresize
from mypath import Path
from dataloaders import make_data_loader
from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap
from dataloaders.datasets import pascal, mpgw
from torch.utils.data import DataLoader


def transform(image):
    return tr.Compose([
        tr.Resize(513),
        tr.CenterCrop(513),
        tr.ToTensor(),
        tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225)),
        ])(image)


def test(args):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    _, val_loader, _, nclass = make_data_loader(args, **kwargs)

    checkpoint = torch.load(args.ckpt)
    if checkpoint is None:
        raise ValueError

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DeepLab(num_classes=nclass,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=True,
                    freeze_bn=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    tbar = tqdm(val_loader)
    num_img_tr = len(val_loader)
    for i, sample in enumerate(tbar):
        x1, x2, y1, y2 = [int(item) for item in sample['img_meta']['bbox_coord']]  # bbox coord
        w, h = x2 - x1, y2 - y1
        img =  sample['img_meta']['image'].squeeze().cpu().numpy()
        img_w, img_h = img.shape[:2]

        inputs = sample['image'].cuda()
        output = model(inputs).squeeze().cpu().numpy()
        pred = np.argmax(output, axis=0)
        result = decode_segmap(pred, dataset=args.dataset, plot=False)

        result = imresize(result, (w, h))
        result_padding = np.zeros(img.shape, dtype=np.uint8)
        result_padding[y1: y2, x1: x2] = result
        result = img // 2 + result_padding * 127
        result[result > 255] = 255
        plt.imsave(os.path.join('run', args.dataset, 'deeplab-resnet', 'output', str(i)), result)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--dataset', type=str, default='pascal')
    parser.add_argument('--base_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_sbd', type=bool, default=False)
    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
   main()
