"""Contains all dataset classes

"""

import pickle
import torch
import sys
from torch.utils.data import Dataset
sys.path.insert(1, '../')
import pickle


class GAN_PKL_DATASET(Dataset):
    '''Dataset for CycleGAN

    Reads '.pkl' file containing processed data, gotten by running '../processing/example.py'
    '''
    def __init__(self, dirpath, genres, normalise, on_off = False, num_segments=0):
        if len(genres) > 2:
            raise Exception('More than two genres specified')

        dset_file = open(dirpath, 'rb')

        dataset = pickle.load(dset_file)
        self.datasetA = dataset[genres[0]]
        self.datasetB = dataset[genres[1]]

        if num_segments < 1:
            maxlen = min(len(self.datasetA), len(self.datasetB))
            self.datasetA = self.datasetA[:maxlen]
            self.datasetB = self.datasetB[:maxlen]
        else:
            self.datasetA = self.datasetA[:num_segments]
            self.datasetB = self.datasetB[:num_segments]

        print(len(self.datasetA), len(self.datasetB))
        self.normalise = normalise
        self.on_off = on_off

    def __len__(self):
        return len(self.datasetA)

    def __getitem__(self, idx):
        segmentA = torch.FloatTensor(self.datasetA[idx])
        segmentB = torch.FloatTensor(self.datasetB[idx])

        if self.normalise:
            segmentA = torch.where(segmentA>0, (segmentA/127)*(1-0.5)+0.5, torch.zeros_like(segmentA))
            segmentB = torch.where(segmentB>0, (segmentB/127)*(1-0.5)+0.5, torch.zeros_like(segmentB))

        if self.on_off:
            segmentA = torch.where(segmentA>0, torch.ones_like(segmentA), torch.zeros_like(segmentA))
            segmentB = torch.where(segmentB>0, torch.ones_like(segmentB), torch.zeros_like(segmentB))


        if segmentA.dim() < 3:
            segmentA = segmentA.unsqueeze(0)
            segmentB = segmentB.unsqueeze(0)
        return segmentA, segmentB
