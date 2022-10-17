import os
from glob import glob
import torch 
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import json
import random
from config import config

num_all = config.number_Num

data_dir = {
    'train_data': './_data/train/mchar_train/',
    'val_data': './_data/val/mchar_val/',
    'test_data': './_data/test/mchar_test/',
    'train_label': './_data/train.json',
    'val_label': './_data/val.json',
    'submit_file': './_data/submit_A.csv'
}

transforms_train =  [
                    transforms.Resize((128, 256)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    ]

transforms_val =    [
                    transforms.Resize((128, 256)),
                    ]

transforms_basic_0 =[
                    transforms.Resize(128),
                    transforms.CenterCrop((128, 224))
                    ]
                        
transforms_orig =   [
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.RandomGrayscale(0.1),
                    transforms.RandomAffine(15, translate=(0.05, 0.1), shear=5)
                    ]

transforms_end =    [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]

class DigitsDataset(Dataset):

    def __init__(self, mode='train', size=(128, 256), trans= True):
        super(DigitsDataset, self).__init__()
        
        modeDict = {'train':1, 'val':1, 'test':0}
        modeDict[mode]
        self.mode = mode
        
        self.width = 224
        self.trans = trans
        self.size = size
        self.batch_count = 0
        if mode == 'test':
            self.imgs = glob(data_dir['test_data'] + '*.png')
            self.labels = None
        else:
            labels = json.load(open(data_dir['%s_label' % mode], 'r'))

            imgs = glob(data_dir['%s_data' % mode] + '*.png')
            self.imgs = [(img, labels[os.path.split(img)[-1]]) for img in imgs \
                         if os.path.split(img)[-1] in labels]

    def __getitem__(self, idx):
        if self.mode != 'test':
            image, label = self.imgs[idx]
        else:
            image = self.imgs[idx]
            label = None
        image = Image.open(image)
        
        transformer = self.getTransformer()
        image_out = transformer(image)
        if self.mode != 'test':
            adder = num_all - len(label['label'])
            # adder > 0
            label = torch.tensor(label['label'][ : num_all] + (adder) * [10]).long()
            return  image_out, label
        else:
            return image_out, self.imgs[idx]

    def getTransformer(self):
        transforms_basic =  [
                            transforms.Resize(128),
                            transforms.CenterCrop((128, self.width))
                            ]
        if self.trans:
            transforms_basic.extend(transforms_orig)
        transforms_basic.extend(transforms_end)
        transformer = transforms.Compose(transforms_basic)
        return transformer
    
    def __len__(self):
        return len(self.imgs)

    # for dataloader parameter: collect_fn
    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == 'train':
            if self.batch_count > 0 and self.batch_count % 10 == 0:
                self.width = random.choice(range(224, 256, 16))
        self.batch_count += 1
        return torch.stack(imgs).float(), torch.stack(labels)
    