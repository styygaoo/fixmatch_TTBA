import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from augumentations.randomaugment import RandAugmentMC
import matplotlib.pyplot as plt

def unpack_and_move(data):
    if isinstance(data, (tuple, list)):
        weak = data[0].to(device, non_blocking=True)
        strong = data[1].to(device, non_blocking=True)
        gt = data[2].to(device, non_blocking=True)
        return weak, strong, gt
    if isinstance(data, dict):
        # print("hier")
        keys = data.keys()
        weak = data['weak'].to(device, non_blocking=True)
        strong = data['strong'].to(device, non_blocking=True)
        gt = data['depth'].to(device, non_blocking=True)
        # print(image.shape)
        # print(gt.shape)
        return weak, strong, gt
    print('Type not supported')

def inverse_depth_norm(depth):
    depth = maxDepth / depth
    depth = torch.clamp(depth, maxDepth / 100, maxDepth)
    return depth

def depth_norm(self, depth):
    depth = torch.clamp(depth, maxDepth / 100, maxDepth)
    depth = maxDepth / depth
    return depth

class CenterCrop(object):
    """
    Wrap torch's CenterCrop
    """
    def __init__(self, output_resolution):
        print(output_resolution)
        self.crop = transforms.CenterCrop(output_resolution)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth)
        image = self.crop(image)
        depth = self.crop(depth)

        return {'image': image, 'depth': depth}




class TransformFixMatch(object):
    def __init__(self):
        self.crop = transforms.CenterCrop((192, 640))

        self.weak = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=(192, 640))])
        self.strong = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=(192, 640)),
            RandAugmentMC(n=2, m=10)])

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        # plt.imshow(image)
        # plt.show()
        # print(np.array(depth))
        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))

        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth)
            # print(np.array(depth))
        weak = self.weak(image)
        # plt.imshow(weak)
        # plt.show()
        strong = self.strong(image)
        # print(np.array(depth))
        depth = self.crop(depth)
        # print(np.array(depth))
        # plt.imshow(strong)
        # plt.show()

        return {'weak': weak, 'strong': strong, 'depth': depth}




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
maxDepth = 80