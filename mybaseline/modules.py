import torch
import torch.nn as nn
import models

class Reshaper(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.bs = batch_size
    
    def __call__(self, X):
        return X.reshape(self.bs, -1)
    pass

class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
                
        model_conv = models.resnet18(pretrained= False, num_classes = 11)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        
        #self.cnn = models.resnet18(num_classes = 11)
        
        self.maxpool = nn.MaxPool2d(2,2)
        self.dp1 = nn.Dropout(0.5)
        num = 512
        
        self.fc1_0 = nn.Linear(num, 11)
        self.fc2_0 = nn.Linear(num, 11)
        self.fc3_0 = nn.Linear(num, 11)
        self.fc4_0 = nn.Linear(num, 11)
        self.fc5_0 = nn.Linear(num, 11)
        
        # Use 11 numbers to predict the probs (1-10)
        self.fc1_1 = nn.Linear(128, 11)
        self.fc2_1 = nn.Linear(128, 11)
        self.fc3_1 = nn.Linear(128, 11)
        self.fc4_1 = nn.Linear(128, 11)
        self.fc5_1 = nn.Linear(128, 11)
        
        
    def forward(self, img):        
        feat = self.cnn(img)
        #feat = self.maxpool(feat)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        #feat = torch.reshape(feat, (40 ,-1))
        feat = self.dp1(feat)
        
        c1 = self.fc1_0(feat)
        c2 = self.fc2_0(feat)
        c3 = self.fc3_0(feat)
        c4 = self.fc4_0(feat)
        """
        c1 = self.fc1_1(c1)
        c2 = self.fc2_1(c2)
        c3 = self.fc3_1(c3)
        c4 = self.fc4_1(c4)
        """
        return c1, c2, c3, c4#, c5