import os
import torch
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import json

from config import config
from SVHNdata import *
from models import *
from utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Main(object):
    def __init__(self, model, val, change, name):
        self.trainer = Trainer(model, val, change, name)
        pass
    def train(self):
        self.trainer.train()
    def eval(self):
        self.trainer.eval()
    def predict(self):
        
        pass
    pass
    


class Trainer(object):
    
    def __init__(self, model= DigitsResnet18(config.class_num),
                 val=True, change= False, modelName = 'resnet18'):

        self.modelName = modelName
        self.device = device

        self.train_set = DigitsDataset(mode='train')
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True,
                                       pin_memory=True, \
                                       drop_last=True, collate_fn=self.train_set.collect_fn,
                                       num_workers= 2)

        if val:
            self.val_loader = DataLoader(DigitsDataset(mode='val', trans=False), batch_size= config.batch_size, \
                                         num_workers=2, pin_memory=True, drop_last=False)
        else:
            self.val_loader = None
        self.val = val
        self.change = change
        
        # self.model = DigitsMobilenet(config.class_num).to(self.device)
        self.model = model.to(device)

        self.criterion = LabelSmoothEntropy().to(self.device)

        # self.optimizer = SGD(self.model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weights_decay, nesterov=True)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                              amsgrad=False)
        
        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=0)
        # self.lr_scheduler
        self.best_acc = config.acc
        self.acc = config.acc

        modelPath = config.pretrained 
        isFile = os.path.isfile(modelPath)
        if config.pretrained is not None and isFile:
            self.load_model(config.pretrained, changed= self.change)
            self.best_acc = config.acc
            print('Load model from %s, Eval Acc: %.2f' % (config.pretrained, config.acc))

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            acc = self.train_epoch(epoch)
            if self.val:
                print('Start Evaluation: ')
                acc = self.eval()
                
            #if acc > self.best_acc:
            os.makedirs(config.checkpoints, exist_ok=True)
            save_path = config.checkpoints + 'epoch-%s-%d-bn-acc-%.2f.pth' % (self.modelName, epoch + 1, acc)
            self.save_model(save_path)
            print('%s saved successfully...' % save_path)
            self.best_acc = acc

    def train_epoch(self, epoch):
        total_loss = 0
        corrects = 0
        tbar = tqdm(self.train_loader)
        tbar.set_description_str(f'epoch {epoch} ')
        self.model.train()
        for i, (img, label) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(img)
            loss = self.criterion(pred[0], label[:, 0]) + \
                   self.criterion(pred[1], label[:, 1]) + \
                   self.criterion(pred[2], label[:, 2]) + \
                   self.criterion(pred[3], label[:, 3]) 
                   
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            temp = torch.stack([ \
                pred[0].argmax(1) == label[:, 0], \
                pred[1].argmax(1) == label[:, 1], \
                pred[2].argmax(1) == label[:, 2], \
                pred[3].argmax(1) == label[:, 3], ], dim=1)

            corrects += torch.all(temp, dim=1).sum().item()
            tbar.set_postfix_str(
                'loss: %.3f, acc: %.3f' % (loss / (i + 1), corrects * 100 / ((i + 1) * config.batch_size)))
            if (i + 1) % config.print_interval == 0:
                self.lr_scheduler.step()
        self.acc = corrects * 100 / ((i + 1) * config.batch_size)
        return self.acc

    def eval(self):
        self.model.eval()
        corrects = 0
        with torch.no_grad():
            tbar = tqdm(self.val_loader)
            for i, (img, label) in enumerate(tbar):
                img = img.to(device)
                label = label.to(device)
                pred = self.model(img)
                # temp = t.stack([])
                temp = torch.stack([
                    pred[0].argmax(1) == label[:, 0], \
                    pred[1].argmax(1) == label[:, 1], \
                    pred[2].argmax(1) == label[:, 2], \
                    pred[3].argmax(1) == label[:, 3], \
                    ], dim=1)

                corrects += torch.all(temp, dim=1).sum().item()
                tbar.set_description('Val Acc: %.2f' % (corrects * 100 / ((i + 1) * config.batch_size)))
        self.model.train()
        self.acc = corrects / (len(self.val_loader) * config.batch_size)
        return self.acc

    def save_model(self, save_path):
        dicts = {}
        dicts['model'] = self.model.state_dict()
        
        torch.save(dicts, save_path)
        
        json_str = json.dumps({'acc':self.acc, 'pretrained':save_path})
        with open('./_mission/test/params.json', 'w') as f:
            f.write(json_str)

    def load_model(self, load_path, changed=False):

        dicts = torch.load(load_path)
        
        # if the model is not resnet18, changed = True
        if not changed:
            self.model.load_state_dict(dicts['model'])

        else:
            dicts = torch.load(load_path)['model']

            keys = list(self.model.state_dict().keys())
            values = list(dicts.values())

            new_dicts = {k: v for k, v in zip(keys, values)}
            try:
                self.model.load_state_dict(new_dicts, strict= False)
            except:
                pass
                
                

if __name__ == '__main__':  
    #print([ele for ele in x if ele != 10])
    # release the cache
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    model = DigitsResnet50()
    #model = DigitsMobilenet()
    model = DigitsResnet18()
    progress = Main(model, True, True, 'resnet18')
    progress.train()
    #trainer = Trainer(val= True)
    #trainer.train()