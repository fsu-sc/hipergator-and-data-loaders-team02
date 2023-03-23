import torch
import torchvision
import matplotlib.pyplot as plt
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms

import itk
import re

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

def resample(img):
    metadataIMG=itk.dict_from_image(img)


    pixdims=[float(metadataIMG['spacing'][0]),float(metadataIMG['spacing'][1]),float(metadataIMG['spacing'][2])]

    dims=[int(metadataIMG['size'][0]),float(metadataIMG['size'][1]),float(metadataIMG['size'][2])]
    
    new_spacing=[2.5, 2.5, 2.5]

    interpolator = itk.LinearInterpolateImageFunction.New(img)

    resampler=itk.ResampleImageFilter.New(img)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(itk.size(img))
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(interpolator)
    resampler.SetSize([int((pixdims[0]/2)*dims[0]),int((pixdims[1]/2)*dims[1]),int((pixdims[2]/2)*dims[2])])
    #resampler.SetSize([100,100,100])
    resampler.Update()
    resampled=resampler.GetOutput()
    return resampled

class MyDataset(Dataset):
    def __init__(self, img_dir, masks_dir, transform=None):
        self.img_dir = img_dir
        self.masks_dir = masks_dir
        #self.imgs_names = os.listdir(img_dir)
        #attempt below to filter to only adc MRI modality
        #self.imgs_names = [f for f in os.listdir(img_dir) if re.match('*adc*', f)]
        self.imgs_names = []
        for file_name in os.listdir(img_dir):
            if 'adc' in file_name:
                self.imgs_names.append(file_name)

        
        self.masks_names = os.listdir(masks_dir)
        self.imgs_len = len(self.imgs_names)
        self.transform = transform

    def __len__(self):
        return self.imgs_len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs_names[idx]).replace("\\","/")
        seg_path = os.path.join(self.masks_dir, self.masks_names[idx]).replace("\\","/")
        image = itk.imread(img_path)
        seg = itk.imread(seg_path)
        #image = read_image(img_path)
        #seg = read_image(seg_path)
        
        #print(image.shape)
        #print(seg.shape)
        #print(self.imgs_names[idx])
        #print(self.masks_names[idx])
        RESimage = resample(image)
        RESseg = resample(seg)
        #print("Post resampling")
        #print(RESimage.shape)
        #print(RESseg.shape)
        
        #imgSliceNum = int(itk.dict_from_image(RESimage)['size'][0])
        #segSliceNum = int(itk.dict_from_image(RESseg)['size'][0])
        
        # resample image and mask to (some values here)
        #if self.transform:
         #   image = self.transform(image)
          #  seg = self.transform(seg)
        #print("NEW version 10 Spacing, Hard Coded Size, Slice Attempt")
        return itk.array_from_image(RESimage).astype(np.float32)[:50, :100, :100], itk.array_from_image(RESseg).astype(np.int16)[:50, :100, :100]
        # return image, seg
       # return itk.array_from_image(image).astype(np.float32), itk.array_from_image(seg).astype(np.int16)[:20, :100, :100]
    
    # P1 .5,.5,.5 sims 10,10,10 -> resample to 1,1,1 dims 5,5,5 -> crop to 0:4, 0:4, 0:4
    # P2 .2,.2,.2 sim 24,24,24  -> resample to 1,1,1, dims 6,6,6 -> crop to 0:4, 0:4, 0:4
    
 