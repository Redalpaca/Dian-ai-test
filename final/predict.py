import pandas as pd
from models import *
from tqdm import tqdm
from config import config
from torch.utils.data import DataLoader
from SVHNdata import *

def predicts(model_paths):
    test_loader = DataLoader(DigitsDataset(mode='test', trans=False),
                             batch_size= config.batch_size, shuffle=False,\
                             num_workers=2, pin_memory=True, drop_last=False)
    results = []
    path1 = './_mission/test/modelSave/epoch-resnet18-2-bn-acc-0.72.pth'
    path2 = './_mission/test/modelSave/epoch-resnet18-2-bn-acc-87.69.pth'
    net1 = DigitsResnet18().cuda()
    net1.load_state_dict(torch.load(path1)['model'])


    net2 = DigitsResnet18().cuda()
    net2.load_state_dict(torch.load(path2)['model'])
    # print('Load model from %s successfully'%model_path)

    tbar = tqdm(test_loader)
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for img, img_names in tbar:
            img = img.cuda()
            #pred = mb_net(img)
            pred = [0.4 * a + 0.6 * b for a, b in zip(net1(img), net2(img))]
            results += [[name, code] for name, code in zip(img_names, parse2class(pred))]

    # result.sort(key=results)
    results = sorted(results, key=lambda x: x[0])

    write2csv(results)
    return results



def parse2class(prediction):
    
    ch1, ch2, ch3, ch4 = prediction

    char_list = [str(i) for i in range(10)]
    char_list.append('')


    ch1, ch2, ch3, ch4 = ch1.argmax(1), ch2.argmax(1), ch3.argmax(1), ch4.argmax(1)

    ch1, ch2, ch3, ch4 = [char_list[i.item()] for i in ch1], [char_list[i.item()] for i in ch2], \
                    [char_list[i.item()] for i in ch3], [char_list[i.item()] for i in ch4] \

    res = [c1+c2+c3+c4 for c1, c2, c3, c4 in zip(ch1, ch2, ch3, ch4)]             
    return res


def write2csv(results):
    """

    results(list):

    """
    #定义输出文件
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(lambda x: x.split('/')[-1])
    save_name = './_mission/test/results.csv'
    df.to_csv(save_name, sep=',', index=None)
    print('Results.saved to %s'%save_name)
    
if __name__ == '__main__':
    import os
    print(os.path.isfile('./_mission/test/results.csv'))
    predicts(True)
    pass
