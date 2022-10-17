"""_summary_
    variable 'config' is used to save the hyper-parameters.
"""
import json
class Config:
    # if use resnet50, bs should not be 32
    # batch_size = 32
    batch_size = 40

    lr = 1e-3
    
    class_num = 11

    # for SGD optim
    momentum = 0.9

    weights_decay = 1e-4

    eval_interval = 1

    checkpoint_interval = 1

    print_interval = 50

    checkpoints = './_mission/test/modelSave/'

    # default file name.
    pretrained = 'epoch-resnet18-1.pth'
    
    # default acc
    acc = 0 
    
    start_epoch = 0

    epoches = 100

    smooth = 0.1

    erase_prob = 0.5
    
    number_Num = 4

config = Config()
with open('./_mission/test/params.json', 'r') as f:
    params = json.loads(f.read())

config.acc = params['acc']
config.pretrained = params['pretrained']