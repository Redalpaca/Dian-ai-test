import torch
import torch.nn as nn
import numpy as np
import modules
from torch.utils import data
import torchvision.transforms as transforms
from dataSet import SVHN_Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

num_epoch = 30
batch = 40

model = modules.SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.96,
                                                last_epoch=-1)

best_loss = 1000.0


transforms_train =  transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    #transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transforms_val =    transforms.Compose([
                    transforms.Resize((64, 128)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device = "cuda" if torch.cuda.is_available() else "cpu"
use_cuda = True if torch.cuda.is_available() else False

def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()
    train_loss = []
    
    pbar = tqdm(train_loader)
    pbar.set_description_str(f'epoch: {epoch : 2}')
    for input, target in pbar:
        
        input = input.to(device)
        target = target.to(device)
            
        c0, c1, c2, c3 = model.forward(input)
        loss =  criterion(c0, target[:, 0]) + \
                criterion(c1, target[:, 1]) + \
                criterion(c2, target[:, 2]) + \
                criterion(c3, target[:, 3]) 
                #criterion(c4, target[:, 4]) 
                #criterion(c5, target[:, 5])
        
        # loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        pbar.set_postfix_str(f'loss: {loss: .2f}')
    scheduler.step()
    #plt.show()
    return np.mean(train_loss)

def validate(val_loader, model, criterion):
    model.eval()
    val_loss = []

    with torch.no_grad():
        pbar = tqdm(val_loader)
        for input, target in pbar:
            
            input = input.to(device)
            target = target.to(device)
            
            c0, c1, c2, c3 = model(input)
            loss = criterion(c0, target[:, 0]) + \
                    criterion(c1, target[:, 1]) + \
                    criterion(c2, target[:, 2]) + \
                    criterion(c3, target[:, 3]) 
                    #criterion(c4, target[:, 4]) #+ \
                    #criterion(c5, target[:, 5])
            # loss /= 6
            val_loss.append(loss.item())
    #fig = plt.subplot()
    #plt.show()
    return np.mean(val_loss)

def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    
    # TTA 次数
    for _ in range(tta):
        test_pred = []
    
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()
                
                c0, c1, c2, c3 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(), 
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(), 
                        c3.data.cpu().numpy(),
                        #c4.data.cpu().numpy(),
                        #c5.data.cpu().numpy()
                        ],axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(), 
                        c1.data.numpy(),
                        c2.data.numpy(), 
                        c3.data.numpy(),
                        #c4.data.numpy(),
                        #c5.data.numpy()
                        ], axis=1)
                
                test_pred.append(output)
        
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta

def main():
    best_loss = 1000.0
    
    train_loader = data.DataLoader(SVHN_Dataset(mode= 'train', transforms= transforms_train), 
                                          batch_size= batch, 
                                          shuffle= True, 
                                          num_workers= 2)
    val_loader = data.DataLoader(SVHN_Dataset(mode= 'val', transforms= transforms_val),
                                        batch_size= batch,
                                        shuffle= False,
                                        num_workers= 2) 
    model = modules.SVHN_Model1()

    if use_cuda:
        model = model.cuda()
    for epoch in range(num_epoch):
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, model, criterion)
        val_label = val_loader.dataset.getNumber()
        val_predict_label = predict(val_loader, model, 1)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),
            #val_predict_label[:, 44:55].argmax(1),
        ]).T
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x!=10])))
        
        val_char_acc = 0
        for i, label_pred in enumerate(val_label_pred):
            end = len(val_label[i])
            if list(label_pred[:end]) == val_label[i]:
                val_char_acc += 1
        #val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
        val_char_acc /= len(val_label)
        
        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
        print('Val Acc', val_char_acc)
        # 记录下验证集精度
        if val_loss < best_loss:
            best_loss = val_loss
            # print('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model.state_dict(), './_mission/modelSave/model.pt')    
        
        """  
        pbar = tqdm(range(400))
        pbar.set_description_str('Sleeping: ')
        for i in pbar:
            time.sleep(1)
        """
        fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
        fig.suptitle('Plot predicted samples')
        plt.show()
    pass

if __name__ == '__main__':
    main()
    pass