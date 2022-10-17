import os
import cv2
import json
import tqdm
def transform2YOLO(mode = 'val'):
    train_json = json.load(open(f'./_data/{mode}.json'))
    pbar = tqdm.tqdm(train_json)
    for x in pbar:
        img=cv2.imread(f"./_data/{mode}/mchar_{mode}/"+x)
        width=img.shape[1]
        height=img.shape[0]
        train_label =list(map(int,train_json[x]['label']))
        train_height=list(map(int,train_json[x]['height']))
        train_left=list(map(int,train_json[x]['left']))
        train_width=list(map(int,train_json[x]['width']))
        train_top=list(map(int,train_json[x]['top']))
        loc_pic=f"./_mission/YOLO/yolov5-master/tianchi/labels/{mode}/"+x.split('.')[0]+'.txt' 
        pic=open(loc_pic,"w")
        for i in range(len(train_label)):
            pic_label=train_label[i]
            pic_x=(train_left[i]+train_width[i]/2)/width
            pic_y=(train_top[i]+train_height[i]/2)/height
            pic_width=train_width[i]/width
            pic_height=train_height[i]/height            
            pic.write(str(pic_label)+" "+str(pic_x)+" "+str(pic_y)+" "+str(pic_width)+" "+str(pic_height))
            pic.write("\n")
        pic.close()

transform2YOLO()