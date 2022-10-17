import torch
import torch.nn as nn
from torch.utils import data
import os
import json
import numpy as np
from PIL import Image

"""Hints

Format of the Labels:
    height, [number], left, top, width

"""

# 最多有 6 个字符，baseline模型只给出了5个回归，还需要扩充。
All = 4

class SVHN_Dataset(data.Dataset):
    
    def __init__(self, root = './_data/', mode = 'train', transforms = None):
        super().__init__()
        # Avoid invalid mode input.
        modeDict = {'train':1, 'val':1, 'test':0}
        modeDict[mode]
        self.mode = mode
        self.transforms = transforms
        # format is like: _data/ train/ mchar_train/ ...jpg
        self.root = root + mode + '/mchar_' + mode + '/'
        self.labelPath = root + mode + '.json' 
        
        if modeDict[mode]:
            with open(self.labelPath, 'r') as f:
                jsonStr = f.read()
            labelsDict = json.loads(jsonStr)
            self.labels = []
            for v in labelsDict.values():
                #self.labels.append(v)
                self.labels.append(torch.tensor(list(v.values())).to(torch.long))
                pass
            
        self.images = os.listdir(self.root)

    def __getitem__(self, index):
        #return super().__getitem__(index)
        img = Image.open(self.root + self.images[index]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
            
        if self.mode == 'test':
            return img
        else:
            lbl = self.labels[index]
            # fill the empty spaces with invalid value: 10
            dim_add = All - lbl.shape[1]
            if dim_add >= 0:
                tensor_add = (torch.zeros(5, dim_add) + 10).to(torch.long) 
                lbl = torch.cat((lbl, tensor_add), dim = 1)
            # 定长字符识别的思路用不上BBOX回归，暂且放置
            return img, lbl[1][:All]
        
    
    def __len__(self):
        return len(self.labels)
        return 2000
    
    def getNumber(self):
        numberLabels = []
        for label in self.labels:
            numberLabels.append(list(map(str, label[1].numpy())))
        return numberLabels
        pass



if __name__ == '__main__':
    #set = SVHNDataset('./_data/train/mchar_train','./_data/train.json')
    set = SVHN_Dataset(mode= 'val')
    #print(set[100])
    labels = set.getNumber()
    print(str(labels[10]))
    str1 = '1234'
    
    pass