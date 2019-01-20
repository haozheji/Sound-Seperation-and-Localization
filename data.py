import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random
import cv2
import numpy as np
import os
import copy
from tqdm import tqdm

import torchvision
from torchvision import transforms
from process_audio import prepare_test
image_trans = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class VisualAudioDataset(Dataset):
    def __init__(self, exs):
        self.exs = exs
    def __len__(self):
        return len(self.exs)
    def __getitem__(self, index):
        return vectorize(self.exs[index])



class SampleDataset(Dataset):
    def __init__(self, exs):
        self.exs = exs
        self.rand_list = []
        for i in range(len(self.exs)):
            for j in range(len(self.exs)):
                if i!=j:
                    self.rand_list.append([i,j])
        self.rand_list = np.random.permutation(self.rand_list)
        #print(self.rand_list)
        #self.rand_list = [[0,1], [0,2], [1,2],[2,0],[1,0],[2,1]]
        #self.rand_list = np.random.permutation(range(len(self)))
    def __len__(self):
        return len(self.rand_list)
    def __getitem__(self, index):
        rand_index = random.sample(range(len(self.exs)), 2)
        #print(rand_index)
        #return vectorize2(self.exs[rand_index[0]], self.exs[rand_index[1]])
        return vectorize2(self.exs[self.rand_list[index][0]], self.exs[self.rand_list[index][1]])

class SequenceDataset(Dataset):
    def __init__(self, exs, test=False):
        self.exs = exs
        self.test = test
    def __len__(self):
        return len(self.exs)
    def __getitem__(self, index):
        if self.test:
            return vectorize_test(self.exs[index])
        else:
            return vectorize_dev(self.exs[index])


def vectorize_test(ex):
    #lstfts = ex['lstft']
    #rstfts = ex['rstft']
    #print(ex.keys())
    #exit()
    mstfts = ex['mstft'][0]
    limg = ex['limg']
    rimg = ex['rimg']
    #pass
    return [[_vectorize(None, None, None, None, limg, rimg, mstft) for mstft in mstfts], ex['mstft'][1], ex['mstft'][2]]

def vectorize_dev(ex):
    lstft, laudio = ex['lstft']
    rstft, raudio = ex['rstft']
    limg = ex['limg']
    rimg = ex['rimg']
    mstft, maudio = ex['mstft']
    return _vectorize(lstft, rstft, laudio, raudio, limg, rimg, mstft)

def vectorize2(ex1, ex2):
    lstft, laudio = ex1['stft']
    rstft, raudio = ex2['stft']
    limg = ex1['image']
    rimg = ex2['image']
    mstft = lstft + rstft
    return _vectorize(lstft, rstft, laudio, raudio, limg, rimg, mstft)

def vectorize(ex):
    lstft, laudio = ex['left_stft']
    rstft, raudio = ex['right_stft']
    limg = ex['left_image']
    rimg = ex['right_image']
    mstft = lstft + rstft
    return _vectorize(lstft, rstft, laudio, raudio, limg, rimg, mstft)

def _vectorize(lstft, rstft, laudio, raudio, limg, rimg, mstft, mode='log'):
    if lstft is not None:
        #ratio_r = (np.abs(rstft) >=  np.abs(lstft)).astype('float')
        ratio_r = (np.abs(rstft) + 1e-30*np.ones((256,256))) / (np.abs(lstft) + np.abs(rstft) + 1e-30*np.ones((256,256)))
        #ratio_l = (np.abs(lstft) >= np.abs(rstft)).astype('float')
        ratio_l = (np.abs(lstft) + 1e-30*np.ones((256,256))) / (np.abs(lstft) + np.abs(rstft) + 1e-30*np.ones((256,256)))
        ratio_r = torch.from_numpy(ratio_r).float()
        ratio_l = torch.from_numpy(ratio_l).float()
        rstft = torch.from_numpy(np.abs(rstft)).float()
        lstft = torch.from_numpy(np.abs(lstft)).float()

    mstft = torch.from_numpy(np.abs(mstft)).float()
    if mode == 'log':
        if lstft is not None:
            lstft = torch.log(lstft.clamp(min=1e-30))
            rstft = torch.log(rstft.clamp(min=1e-30))
        mstft = torch.log(mstft.clamp(min=1e-30))

    limg = [cv2.resize(limg[i], dsize=(224, 224), interpolation=cv2.INTER_CUBIC) for i in range(limg.shape[0])]
    limg = np.stack(limg, axis=0)# size: T x 224 x 224 x 3

    limg_t = torch.randn(limg.shape[0], limg.shape[3], limg.shape[1], limg.shape[2]).float()


    for t in range(limg.shape[0]):
        limg_t[t] = image_trans(limg[t])
    # out limg: T x 3 x 224 x 224

    #Mean = np.mean(limg, axis=(1,2,3))
    #Std = np.std(limg, axis=(1,2,3))
    #limg = (limg - Mean) / np.sqrt(Std)
    # limg = torch.from_numpy(limg).float()
    rimg = [cv2.resize(rimg[i], dsize=(224, 224), interpolation=cv2.INTER_CUBIC) for i in range(rimg.shape[0])]
    rimg = np.stack(rimg, axis=0)
    rimg_t = torch.randn(rimg.shape[0], rimg.shape[3], rimg.shape[1], rimg.shape[2]).float()

    for t in range(rimg.shape[0]):
       rimg_t[t] = image_trans(rimg[t])


    #Mean = np.mean(rimg, axis=(1,2,3))
    #Std = np.std(rimg, axis=(1,2,3))
    #rimg = (rimg - Mean) / np.sqrt(Std)
    # rimg = torch.from_numpy(rimg).float()
    if lstft is not None:
        return mstft, ratio_r, ratio_l, rimg_t, limg_t, rstft, lstft, raudio, laudio
    else:
        return mstft, rimg_t, limg_t

def batchify(batch):
    #print(len(batch))
    #print(len(batch[0]))
    #exit()
    if len(batch[0][0][0]) == 3:
        return batch
    mstfts = torch.stack([x[0] for x in batch], dim=0)
    rmasks = torch.stack([x[1] for x in batch], dim=0)
    lmasks = torch.stack([x[2] for x in batch], dim=0)
    #for r in batch:
    #    print(r[3].size())
    rimgs = torch.stack([x[3] for x in batch], dim=0)

    limgs = torch.stack([x[4] for x in batch], dim=0)
    # used for validation
    rstfts = torch.stack([x[5] for x in batch], dim=0)
    lstfts = torch.stack([x[6] for x in batch], dim=0)
    laudios = [x[7] for x in batch]
    raudios = [x[8] for x in batch]
    return mstfts, rmasks, lmasks, rimgs, limgs, rstfts, lstfts, laudios, raudios

def load_test(args, path):
    loader = prepare_test(path)
    exs = []
    for i, e in enumerate(loader):
        exs.append(e)
        if i == args.load_num:
            break
    return exs

def load_data(args, path, mode= 'train'):
    files = os.listdir(path)
    exs = []
    if mode == 'train':
        for f in tqdm(files[:args.load_num]):
            with np.load(os.path.join(path, f)) as fp:
                temp = {}
                temp['image'] = fp['image']
                temp['stft'] = fp['stft']
            exs.append(temp)

    elif mode == 'dev':
        for f in tqdm(files[:args.load_num]):
            with np.load(os.path.join(path, f)) as fp:
                temp = {}
                temp['limg'] = fp['limg']
                temp['rimg'] = fp['rimg']
                try:
                    temp['rstft'] = fp['rstft']
                    temp['lstft'] = fp['lstft']
                except:
                    pass
                temp['mstft'] = fp['mstft']
            exs.append(temp)
    else:
        raise RuntimeError('load train data or dev data?')
    return exs


def load_pro_data(args, path):
    files = os.listdir(path)[args.load_start:args.load_start + args.load_num]
    exs = []
    for f in tqdm(files):
        with np.load(os.path.join(path, f)) as fp:
            temp = {}
            temp['left_stft'] = fp['left_stft']
            temp['right_stft'] = fp['right_stft']
            temp['left_image'] = fp['left_image']
            temp['right_image'] = fp['right_image']
        exs.append(temp)
    return exs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
