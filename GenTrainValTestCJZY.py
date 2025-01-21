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

img_path = r'M:\\UisYangtzeDataset\\images\\'
img_list = generator_list_of_imagepath(img_path)
print(img_list)
# SH_list = ['000000_SH_json', '000001_SH_json', '000002_SH_json', '000003_SH_json', '000004_SH_json', '000005_SH_json', '000006_SH_json', '000007_SH_json', '000008_SH_json', '000009_SH_json', '001000_SH_json', '001001_SH_json', '001002_SH_json', '001003_SH_json', '001004_SH_json', '001005_SH_json']
#
# test_SH = ['000001_SH_json', '000005_SH_json', '001000_SH_json', '001003_SH_json']  # 2 6 11 14 #
# train_SH = []
# for item in SH_list:
#     if item not in test_SH:
#         train_SH.append(item)
# print(train_SH, test_SH, len(train_SH), len(test_SH))
#
# BJ_list = ['000010_BJ_json', '000011_BJ_json', '000012_BJ_json', '000013_BJ_json', '000014_BJ_json', '000015_BJ_json', '000016_BJ_json', '000017_BJ_json', '000018_BJ_json', '000019_BJ_json', '001010_BJ_json', '001011_BJ_json', '001012_BJ_json', '001013_BJ_json', '001014_BJ_json', '001015_BJ_json', '001016_BJ_json']
# test_BJ = ['000010_BJ_json', '000013_BJ_json', '000018_BJ_json', '001016_BJ_json']  # 1 4 9 17
# train_BJ = []
#
# for item in BJ_list:
#     if item not in test_BJ:
#         train_BJ.append(item)
# print(train_BJ, test_BJ, len(train_BJ), len(test_BJ))
# # id = [i for i in range(225)]
X_train, X_test = train_test_split(img_list, test_size=0.95, random_state=1)

# X_test, X_val = train_test_split(X_test, test_size=0.5, random_state=1) # 0.25 x 0.8 = 0.2

print(len(X_train), len(X_test))


train_txt='data\\trainUV_CJZY_0.5.txt'
# val_txt='data\\val_1_10.txt'
test_txt='data\\testUV_CJZY_9.5.txt'  # 路径1 路径2 路径3 路径4 路径5 标签  hrs,lrs,hpi,sv,pois,label
Trainfile = open(train_txt, "w", encoding='utf-8')
for i in range(len(X_train)):
    X_train_hrs_path = r'M:\\UisYangtzeDataset\\images\\' + str(X_train[i]) + '.tif'
    label = r'M:\\UisYangtzeDataset\\labels\\' + str(X_train[i]) + '.png'
    str1 = X_train_hrs_path + '\t' + label
    Trainfile.write(str1)
    Trainfile.write('\n')
Trainfile.close()

Testfile = open(test_txt, "w", encoding='utf-8')
for i in range(len(X_test)):
    X_test_hrs_path = r'M:\\UisYangtzeDataset\\images\\' + str(X_test[i]) + '.tif'
    label =  r'M:\\UisYangtzeDataset\\labels\\' + str(X_test[i]) + '.png'
    str1 = X_test_hrs_path + '\t' + label
    Testfile.write(str1)
    Testfile.write('\n')
Testfile.close()
