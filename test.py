import argparse
import os
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as tr
import matplotlib.pyplot as plt

from mypath import Path
from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap


def transform(image):
    return tr.Compose([
        tr.Resize(513),
        tr.CenterCrop(513),
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])(image)


def test(ckpt, dataset):
    if dataset == 'pascal':
        data_path = Path.db_root_dir(dataset)
        img_list = open(os.path.join(data_path, 'ImageSets', 'Segmentation', 'val.txt'), 'r').readlines()
    else:
        raise NotImplementedError

    checkpoint = torch.load(ckpt)
    if checkpoint is None:
        raise ValueError

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = DeepLab(num_classes=21,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=True,
                    freeze_bn=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    for img in img_list:
        img = img.replace('\n', '.jpg')
        image = Image.open(os.path.join(data_path, 'JPEGImages', img))
        inputs = transform(image).to(device)
        output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
        pred = np.argmax(output, axis=0)
        result = decode_segmap(pred, dataset="pascal", plot=False)
        plt.imsave(os.path.join('run', 'pascal', 'deeplab-resnet', 'output', img), result)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--dataset', type=str, default='pascal')
    args = parser.parse_args()
    test(args.ckpt, args.dataset)


if __name__ == "__main__":
   main()
