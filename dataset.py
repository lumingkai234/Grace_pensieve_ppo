import os
import torch
import logging
import cv2
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
from augmentation import random_flip, random_crop_and_pad_image_and_labels
from torch.utils.data import DataLoader

class DataSet(data.Dataset):
    def __init__(self, path="/data/dh/Vimeo90k/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        
        out_channel_N = 64
        out_channel_M = 96
        out_channel_mv = 128

        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir="/data/dh/Vimeo90k/vimeo_septuplet/sequences", filefolderlist="/data/dh/Vimeo90k/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()
        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input += [y]
            refnumber = int(y[-5:-4]) - 2
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]
    
        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        input_image, ref_image = random_flip(input_image, ref_image)

        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        return input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv

def get_train_loader(train_set, gpu_num, gpu_per_batch):
    train_loader = DataLoader(dataset = train_set, shuffle=True, num_workers=gpu_num, 
                              batch_size=gpu_per_batch, pin_memory=True)
    return train_loader

if __name__ == '__main__':
    

    train_set = DataSet()
    train_set.get_vimeo()
    print(len(train_set))
    print('Loading train loader...')
    train_loader = get_train_loader(train_set, 4, 4)
    print('Done')
    print(len(train_loader))
    for data in train_loader:
        print(data[0].size())
        print(data[1].size())
        print(data[2].size())
        print(data[3].size())
        print(data[4].size())
        break