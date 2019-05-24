from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import torch
import os
import random

class dataset_pipeline(Dataset):
    def __init__(self,path):
        super(dataset_pipeline, self).__init__()

        self.mfcc = h5py.File(path, 'r')['spec']
        self.label = h5py.File(path, 'r')['label']
        N,D,F=self.mfcc.shape
        self.len=N
       
    def __getitem__(self, index):
        mfcc_item = torch.from_numpy(self.mfcc[index].astype(np.float32))
        label_item = torch.from_numpy(self.label[index].astype(np.float32))
        length_item = torch.from_numpy(np.array(161))
        return mfcc_item, label_item,length_item
    
    def __len__(self):
        return self.len

