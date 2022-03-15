import torch
import os
import pickle
from torch.utils.data import Dataset
from dataloader import DataLoader
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import (RandomFlip,
                                        RandomRotate,
                                        RandomTranslate,
                                        Compose,
                                        Center)

path = '../data/mice_features2/all'
labels = '../data/mice_features2/total_labels.pkl'

class GINDataset(Dataset):

    def __init__(self, root=path, label=labels, phase='train'):
        'Initialization'
        # root = os.path.join(root, phase)
        total = os.listdir(root)
        self.root = root
        self.data = total
        self.phase = phase
        np.random.seed(16)
        with open(label, 'rb') as f:
            self.labels = pickle.load(f)

        total_size = len(total)
        permu = np.random.permutation(total_size)
        if phase == 'train':
            self.list_IDs = permu[:int(total_size*0.9)]
        elif phase == 'val':
            self.list_IDs = permu[int(total_size*0.9):]
        elif phase == 'test':
            self.list_IDs = permu[int(total_size*0.9):]
        else:
            raise ValueError('wrong phase!')
        self.transform = Compose([RandomFlip(0),
                                  RandomFlip(1),
                                  RandomFlip(2),
                                  RandomRotate(360, 0),
                                  RandomRotate(360, 1),
                                  RandomRotate(360, 2),
                                  Center()])
        self.trans = Center()

    def __len__(self):
        'Denotes the total number of samples'
        return self.list_IDs.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        path = self.data[index]
        # label = convert_y_unit(label, 'nM', 'p')
        path = os.path.join(self.root, path)
        with open(path, 'rb') as f:
            x, edge_index, edge_attr = pickle.load(f)
        label = self.labels[path.split('/')[-1].split('.')[0]]
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pos=x[:, -3:])
        if self.phase == 'train':
            self.transform(data)
        else:
            self.trans(data)
        return data, torch.tensor(label).float()

def get_gin_dataloader(path, label_path, batch_size,
                       num_workers=6, phase='train'):
    dataset = GINDataset(path, label_path, phase=phase)
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle)
    return dataloader
