# coding=utf-8
import os
import cv2
import random
import numpy
import sys
from sklearn.model_selection import train_test_split

def generator_list_of_imagepath(path):
    image_list = []
    for image in os.listdir(path):
        # print(path)
        # print(image)
        if 'tif' == image.split('.')[-1]:
            image_list.append(image.split('.')[0])
        # image_list.append(image)
    return image_list

img_path = r'M:\\UV-China\\CJZY城中村置信度0.5-0.8数据集_20231107\\label_08\\'
img_list = generator_list_of_imagepath(img_path)
print(img_list)

train_txt='data\\trainUV_Semantic_CJZY_Semi_0.8.txt'
# val_txt='data\\val_1_10.txt'
# test_txt='data\\testUV_Semantic_CJZY.txt'  # 路径1 路径2 路径3 路径4 路径5 标签  hrs,lrs,hpi,sv,pois,label
Trainfile = open(train_txt, "w", encoding='utf-8')
for i in range(len(img_list)):
    X_train_hrs_path = 'M:\\UV-China\\CJZY城中村置信度0.5-0.8数据集_20231107\\image\\' + str(img_list[i]) + '.tif'
    label = 'M:\\UV-China\\CJZY城中村置信度0.5-0.8数据集_20231107\\label_08\\' + str(img_list[i]) + '.tif'
    str1 = X_train_hrs_path + '\t' + label
    Trainfile.write(str1)
    Trainfile.write('\n')
Trainfile.close()
