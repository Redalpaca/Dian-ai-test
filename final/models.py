import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnet50, resnet34, resnet18
import torchvision

class DigitsMobilenet(nn.Module):
    
  def __init__(self, class_num=11):
    super(DigitsMobilenet, self).__init__()

    self.net = mobilenet_v2(weights= torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).features
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.bn = nn.BatchNorm1d(1280)
    self.fc1 = nn.Linear(1280, class_num)
    self.fc2 = nn.Linear(1280, class_num)
    self.fc3 = nn.Linear(1280, class_num)
    self.fc4 = nn.Linear(1280, class_num)


  def forward(self, img):
    features = self.avgpool(self.net(img)).view(-1, 1280)
    features = self.bn(features)

    fc1 = self.fc1(features)
    fc2 = self.fc2(features)
    fc3 = self.fc3(features)
    fc4 = self.fc4(features)

    return fc1, fc2, fc3, fc4


class DigitsResnet18(nn.Module):

  def __init__(self, class_num=11):
    super(DigitsResnet18, self).__init__()
    self.net = resnet18(weights= torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    self.net.fc = nn.Identity()
    
    self.bn = nn.BatchNorm1d(512)
    self.fc1 = nn.Linear(512, class_num)
    self.fc2 = nn.Linear(512, class_num)
    self.fc3 = nn.Linear(512, class_num)
    self.fc4 = nn.Linear(512, class_num)

  def forward(self, img):
    features = self.net(img).squeeze()

    fc1 = self.fc1(features)
    fc2 = self.fc2(features)
    fc3 = self.fc3(features)
    fc4 = self.fc4(features)

    return fc1, fc2, fc3, fc4


class DigitsResnet50(nn.Module):
    def __init__(self, class_num= 11):
        super(DigitsResnet50, self).__init__()

        # resnet18
        self.net = resnet50(weights= torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
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
    
