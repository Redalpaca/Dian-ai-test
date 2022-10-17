import os
from glob import glob
import torch as t
from PIL import Image
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import torch.nn.functional as F
import json
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnet50, resnet34,resnet18
from torchsummary import summary
import random
from torchvision.models.mobilenet import MobileNetV2

#超参数设定
class Config:

  batch_size = 32

  lr = 1e-3

  momentum = 0.9

  weights_decay = 1e-4

  class_num = 11

  eval_interval = 1

  checkpoint_interval = 1

  print_interval = 50

  checkpoints = './baseline/modelSave/'#这个需要自己创建一个文件夹用来储存权重

  pretrained = './baseline/modelSave/epoch-resnet18-52-bn-acc-73.86.pth' #'/content/NDataset/checkpionts/epoch-mobeilenet-91-bn-acc-72.23.pth'

  start_epoch = 52

  epoches = 100

  smooth = 0.1

  erase_prob = 0.5

config = Config()

data_dir = {
    'train_data': './_data/train/mchar_train/',
    'val_data': './_data/val/mchar_val/',
    'test_data': './_data/test/mchar_test/',
    'train_label': './_data/train.json',
    'val_label': './_data/val.json',
    'submit_file': './_data/submit_A.csv'
}

class DigitsDataset(Dataset):
    """

    DigitsDataset

    Params:
      data_dir(string): data directory

      label_path(string): label path

      aug(bool): wheather do image augmentation, default: True
    """

    def __init__(self, mode='train', size=(128, 256), aug=True):
        super(DigitsDataset, self).__init__()

        self.aug = aug
        self.size = size
        self.mode = mode
        self.width = 224
        # self.weights = np.array(list(label__nums_summary().values()))
        # self.weights = (np.sum(self.weights) - self.weights) / self.weights
        # self.weights = t.from_numpy(self.weights.to_list().append(1)).float()
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
            img, label = self.imgs[idx]
        else:
            img = self.imgs[idx]
            label = None
        img = Image.open(img)
        trans0 = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        min_size = self.size[0] if (img.size[1] / self.size[0]) < ((img.size[0] / self.size[1])) else self.size[1]
        trans1 = [
            transforms.Resize(128),
            transforms.CenterCrop((128, self.width))
        ]
        if self.aug:
            trans1.extend([
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.RandomGrayscale(0.1),
                transforms.RandomAffine(15, translate=(0.05, 0.1), shear=5)
            ])
        trans1.extend(trans0)
        if self.mode != 'test':
            return transforms.Compose(trans1)(img), t.tensor(
                label['label'][:4] + (4 - len(label['label'])) * [10]).long()
        else:
            # trans1.append(transforms.RandomErasing(scale=(0.02, 0.1)))
            return transforms.Compose(trans1)(img), self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == 'train':
            if self.batch_count > 0 and self.batch_count % 10 == 0:
                self.width = random.choice(range(224, 256, 16))

        self.batch_count += 1
        return t.stack(imgs).float(), t.stack(labels)
    
    
class DigitsMobilenet(nn.Module):
    
  def __init__(self, class_num=11):
    super(DigitsMobilenet, self).__init__()

    self.net = mobilenet_v2(pretrained=True).features
    # self.net.classifier = nn.Identity()
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.bn = nn.BatchNorm1d(1280)
    self.fc1 = nn.Linear(1280, class_num)
    self.fc2 = nn.Linear(1280, class_num)
    self.fc3 = nn.Linear(1280, class_num)
    self.fc4 = nn.Linear(1280, class_num)
    # self.fc5 = nn.Linear(1280, class_num)

  def forward(self, img):
    features = self.avgpool(self.net(img)).view(-1, 1280)
    features = self.bn(features)

    fc1 = self.fc1(features)
    fc2 = self.fc2(features)
    fc3 = self.fc3(features)
    fc4 = self.fc4(features)
    # fc5 = self.fc5(features)

    return fc1, fc2, fc3, fc4


class DigitsResnet18(nn.Module):

  def __init__(self, class_num=11):
    super(DigitsResnet18, self).__init__()
    self.net = resnet18(pretrained= False)
    self.net.fc = nn.Identity()
    # self.net = nn.Sequential(
    #     MobileNetV2(num_classes=class_num).features,
    #     nn.AdaptiveAvgPool2d((1, 1))
    # )
    self.bn = nn.BatchNorm1d(512)
    self.fc1 = nn.Linear(512, class_num)
    self.fc2 = nn.Linear(512, class_num)
    self.fc3 = nn.Linear(512, class_num)
    self.fc4 = nn.Linear(512, class_num)
    # self.fc5 = nn.Linear(512, class_num)

  def forward(self, img):
    features = self.net(img).squeeze()

    fc1 = self.fc1(features)
    fc2 = self.fc2(features)
    fc3 = self.fc3(features)
    fc4 = self.fc4(features)
    # fc5 = self.fc5(features)

    return fc1, fc2, fc3, fc4


class DigitsResnet50(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsResnet50, self).__init__()

        # resnet18
        self.net = resnet50(pretrained= False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(*list(self.net.children())[:-1])  # 去除最后一个fc layer
        self.cnn = self.net

        self.hd_fc1 = nn.Linear(2048, 128)  #512改成2048
        self.hd_fc2 = nn.Linear(2048, 128)
        self.hd_fc3 = nn.Linear(2048, 128)
        self.hd_fc4 = nn.Linear(2048, 128)
        # self.hd_fc5 = nn.Linear(512, 128)
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.25)
        # self.dropout_5 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 11)
        self.fc2 = nn.Linear(128, 11)
        self.fc3 = nn.Linear(128, 11)
        self.fc4 = nn.Linear(128, 11)
        # self.fc5 = nn.Linear(128, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        # feat = self.net(img).squeeze()

        feat1 = self.hd_fc1(feat)
        feat2 = self.hd_fc2(feat)
        feat3 = self.hd_fc3(feat)
        feat4 = self.hd_fc4(feat)
        # feat5 = self.hd_fc5(feat)
        feat1 = self.dropout_1(feat1)
        feat2 = self.dropout_2(feat2)
        feat3 = self.dropout_3(feat3)
        feat4 = self.dropout_4(feat4)
        # feat5 = self.dropout_5(feat5)

        c1 = self.fc1(feat1)
        c2 = self.fc2(feat2)
        c3 = self.fc3(feat3)
        c4 = self.fc4(feat4)
        # c5 = self.fc5(feat5)

        return c1, c2, c3, c4
    
class LabelSmoothEntropy(nn.Module):
    def __init__(self, smooth=0.1, class_weights=None, size_average='mean'):
        super(LabelSmoothEntropy, self).__init__()
        self.size_average = size_average
        self.smooth = smooth

        self.class_weights = class_weights

    def forward(self, preds, targets):

        lb_pos, lb_neg = 1 - self.smooth, self.smooth / (preds.shape[0] - 1)

        smoothed_lb = t.zeros_like(preds).fill_(lb_neg).scatter_(1, targets[:, None], lb_pos)

        log_soft = F.log_softmax(preds, dim=1)

        if self.class_weights is not None:
            loss = -log_soft * smoothed_lb * self.class_weights[None, :]

        else:
            loss = -log_soft * smoothed_lb

        loss = loss.sum(1)
        if self.size_average == 'mean':
            return loss.mean()

        elif self.size_average == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError

class Trainer:
    
    def __init__(self, val=True):

        self.device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

        self.train_set = DigitsDataset(mode='train')
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True,
                                       pin_memory=True, \
                                       drop_last=True, collate_fn=self.train_set.collect_fn,
                                       num_workers= 2)

        if val:
            self.val_loader = DataLoader(DigitsDataset(mode='val', aug=False), batch_size=config.batch_size, \
                                         num_workers=2, pin_memory=True, drop_last=False)
        else:
            self.val_loader = None

        #可以切换使用resnet或者Mobilenet
        # self.model = DigitsMobilenet(config.class_num).to(self.device)
        self.model = DigitsResnet18(config.class_num).to(self.device)

        self.criterion = LabelSmoothEntropy().to(self.device)

        #optimizer可选择SGD或者Adam
        # self.optimizer = SGD(self.model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weights_decay, nesterov=True)
        from torch.optim import Adam
        # self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                              amsgrad=False)
        
        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=0)
        # self.lr_scheduler = (self.optimizer, [10, 20, 30], 0.5)
        self.best_acc = 0

    #是否载入预训练模型
        isFile = os.path.isfile(config.pretrained)
        if config.pretrained is not None and isFile:
            self.load_model(config.pretrained)
            # print('Load model from %s'%config.pretrained)
            if self.val_loader is not None:
                acc = self.eval()
            self.best_acc = acc
            print('Load model from %s, Eval Acc: %.2f' % (config.pretrained, acc * 100))

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            acc = self.train_epoch(epoch)
            if (epoch + 1) % config.eval_interval == 0:
                print('Start Evaluation')
                if self.val_loader is not None:
                    acc = self.eval()
                    
                if acc > self.best_acc:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = config.checkpoints + 'epoch-resnet18-%d-bn-acc-%.2f.pth' % (epoch + 1, acc * 100)
                    self.save_model(save_path)
                    print('%s saved successfully...' % save_path)
                    self.best_acc = acc

    def train_epoch(self, epoch):
        total_loss = 0
        corrects = 0
        tbar = tqdm(self.train_loader)
        self.model.train()
        for i, (img, label) in enumerate(tbar):
            # print(img.shape)
            img = img.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(img)
            loss = self.criterion(pred[0], label[:, 0]) + \
                   self.criterion(pred[1], label[:, 1]) + \
                   self.criterion(pred[2], label[:, 2]) + \
                   self.criterion(pred[3], label[:, 3]) \
                # + self.criterion(pred[4], label[:, 4])
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            temp = t.stack([ \
                pred[0].argmax(1) == label[:, 0], \
                pred[1].argmax(1) == label[:, 1], \
                pred[2].argmax(1) == label[:, 2], \
                pred[3].argmax(1) == label[:, 3], ], dim=1)

            corrects += t.all(temp, dim=1).sum().item()
            tbar.set_description(
                'loss: %.3f, acc: %.3f' % (loss / (i + 1), corrects * 100 / ((i + 1) * config.batch_size)))
            if (i + 1) % config.print_interval == 0:
                self.lr_scheduler.step()
        return corrects * 100 / ((i + 1) * config.batch_size)

    def eval(self):
        self.model.eval()
        corrects = 0
        with t.no_grad():
            tbar = tqdm(self.val_loader)
            for i, (img, label) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)
                pred = self.model(img)
                # temp = t.stack([])
                temp = t.stack([
                    pred[0].argmax(1) == label[:, 0], \
                    pred[1].argmax(1) == label[:, 1], \
                    pred[2].argmax(1) == label[:, 2], \
                    pred[3].argmax(1) == label[:, 3], \
                    ], dim=1)

                corrects += t.all(temp, dim=1).sum().item()
                tbar.set_description('Val Acc: %.2f' % (corrects * 100 / ((i + 1) * config.batch_size)))
        self.model.train()
        return corrects / (len(self.val_loader) * config.batch_size)

    def save_model(self, save_path, save_opt=False, save_config=False):
        dicts = {}
        dicts['model'] = self.model.state_dict()
        if save_opt:
            dicts['opt'] = self.optimizer.state_dict()

        if save_config:
            dicts['config'] = {s: config.__getattribute__(s) for s in dir(config) if not s.startswith('_')}

        t.save(dicts, save_path)

    def load_model(self, load_path, changed=False, save_opt=False, save_config=False):

        dicts = t.load(load_path)
        if not changed:
            self.model.load_state_dict(dicts['model'])

        else:
            dicts = t.load(load_path)['model']

            keys = list(self.model.state_dict().keys())
            values = list(dicts.values())

            new_dicts = {k: v for k, v in zip(keys, values)}
            self.model.load_state_dict(new_dicts)

        if save_opt:
            self.optimizer.load_state_dict(dicts['opt'])

        if save_config:
            for k, v in dicts['config'].items():
                config.__setattr__(k, v)
if __name__ == '__main__':     
    trainer = Trainer()
    trainer.train()