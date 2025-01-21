# -*- coding:utf-8 -*-
import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2
import torch.nn.functional as F
import time
from osgeo import gdal
from math import ceil
from PIL import Image
from tifffile import *
import torch.nn as nn
# from dataloader import VaihingenDataset
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate
from dataset import UVdataset
# from baseline_models import FCN, deeplabv3
from Segformer import Segformer_baseline
# from lovasz_losses import Lovasz_softmax
# from heads.fpn import FPN
# from heads.fapn import FaPN
# from heads.upernet import UPerNet
# from HRNet import HRnet
# from densecrf import DenseCRF
from PIL import Image
import matplotlib.pyplot as plt

# from model_fuse import SOST, deeplabv3_resnet50, SOST_RGB, fcn_resnet50, FPN, UNet_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imgsize = 465
class OS:

    def __init__(self, model_path):

        if torch.cuda.is_available():
            self.model = torch.load(model_path).to(device)
        else:
            self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((imgsize, imgsize)),  # 将图像转化为128 * 128
            transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化
        ])


    def detect(self, image):

        image = self.transform(image)
        image = image.to(device)
        image = image.unsqueeze(0)

        # mmdata = self.transform(mmdata)
        # mmdata = mmdata.to(device)
        # mmdata = mmdata.unsqueeze(0)
        #
        # mmdata = mmdata.clone().detach().float()
        image = image.clone().detach().float()
        # print('image.shape, mmdata.shape', image.shape, mmdata.shape)

        _, outputs = self.model(image)

        # image = np.uint8(255 * image.squeeze(0).data.cpu().numpy())
        # image = image.transpose(1, 2, 0)
        # # print(image.shape)
        # outputs = DenseCRF()(image, outputs.squeeze(0).data.cpu().numpy())
        # # print(type(outputs2))
        # outputs = torch.from_numpy(outputs).unsqueeze(0).to(device)

        # print('outputs:', outputs.shape)
        prob = F.softmax(outputs, dim=1)
        # # print('prob:', prob[0])
        # pred = torch.argmax(prob, dim=1)
        # pred = pred.numpy()
        return prob[0]


# def predict_sliding(model1, image, tile_size, classes, overlap):
#     # interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
#     image_size = image.shape
#     # print(image_size)
#     predict_proprely = model1.detect(image)
#     predict_proprely = predict_proprely.cpu().data.numpy()
#     return predict_proprely

def predict_sliding(model1, image, tile_size, classes, overlap):
    # interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    image_size = image.shape
    print(image_size)
    # print('scale size', tile_size)
    # overlap = 1/10 #每次滑动的重合率为1/3
    stride_rows = int(ceil(tile_size[0] * (1 - overlap)))
    stride_cols = int(ceil(tile_size[1] * (1 - overlap)))
    tile_rows = int(ceil((image_size[0] - tile_size[0]) / stride_rows) + 1)  #行滑动步数:
    tile_cols = int(ceil((image_size[1] - tile_size[1]) / stride_cols) + 1)  #列滑动步数:
    print("Need %i x %i prediction tiles @ tile_rows %i and tile_cols %ipx" % (tile_rows, tile_cols, tile_rows, tile_cols))
    full_probs = np.zeros((classes, image_size[0], image_size[1]))  #初始化全概率矩阵
    count_predictions = np.zeros((classes, image_size[0], image_size[1]))   #初始化计数矩阵
    tile_counter = 0    #滑动计数0
    for row in range(tile_rows):    # row = 0,1
        if row % 100 == 0:
            print('in row:', row)
        for col in range(tile_cols):    # col = 0,1,2,3
            x1 = int(col * stride_cols)  #起始位置x1 = 0 * 513 = 0
            y1 = int(row * stride_rows)  #        y1 = 0 * 513 = 0
            x2 = min(x1 + tile_size[1], image_size[1])  #末位置x2 = min(0+769, 2048)
            y2 = min(y1 + tile_size[0], image_size[0])  #      y2 = min(0+769, 1024)
            x1 = max(int(x2 - tile_size[1]), 0)  #重新校准起始位置x1 = max(769-769, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  #y1 = max(769-769, 0)
            # print(x1, x2, y1, y2)
            img = image[y1:y2, x1:x2, :] #滑动窗口对应的图像 imge[:, :, 0:769, 0:769]
            # print('img.shape, mmd.shape:', img.shape, mmdata[y1:y2, x1:x2, :].shape)
            predict_proprely = model1.detect(img)
            # print(predict_proprely.shape)
            # predict_proprely2 = model2.detect(img)
            # predict_proprely3 = model3.detect(img)

            # predict_proprely = 0.33*predict_proprely1 + 0.34*predict_proprely2 + 0.33*predict_proprely3
            predict_proprely = predict_proprely.cpu().data.numpy()
            count_predictions[:, y1:y2, x1:x2] += 1    #窗口区域内的计数矩阵加1
            full_probs[:, y1:y2, x1:x2] += predict_proprely  #窗口区域内的全概率矩阵叠加预测结果
            tile_counter += 1   #计数加1
    # print(count_predictions)
    full_probs /= count_predictions
    return full_probs

def color_predicts(img):
    # img = cv2.imread(label_path,cv2.CAP_MODE_GRAY)
    # label_dict = {'住宅区': 0, '公共服务区域': 1, '商业区': 2, '工业区': 3}
    color = np.zeros([img.shape[0], img.shape[1], 3])  # BGR
    color_ad = np.zeros([img.shape[0], img.shape[1], 3])
    # color[img==0] = [0, 255, 255] # 黄色 裸地
    color[img==0] = [0, 0, 0] #青色
    color[img==1] = [0, 128, 0] #  红色


    return color

def bgr2gray(img):
    color = np.zeros([img.shape[0], img.shape[1]])  # RGB
    """label information of 15 classes:
    industrial land
    RGB:    200,    0,    0
    urban residential
    RGB:    250,    0, 150
    rural residential
    RGB:    200, 150, 150
    traffic land
    RGB:    250, 150, 150
    paddy field
    RGB:    0,     200,    0

    irrigated land
    RGB:    150,  250,   0
    dry cropland
    RGB:    150, 200, 150
    garden plot
    RGB:    200,     0, 200
    arbor woodland
    RGB:    150,     0, 250
    shrub land
    RGB:    150,  150, 250

    natural grassland
    RGB:    250,  200,    0
    artificial grassland
    RGB:    200,  200,    0
    river
    RGB:    0,         0, 200
    lake
    RGB:    0,     150, 200
    pond
    RGB:    0,     200, 250
    """
    color[np.all(img==[0, 0, 0], axis=2)] = 0

    color[np.all(img==[200, 0, 0], axis=2)] = 1
    color[np.all(img==[250, 0, 150], axis=2)] = 2
    color[np.all(img==[200, 150, 150], axis=2)] = 3
    color[np.all(img==[250, 150, 150], axis=2)] = 4
    color[np.all(img==[0, 200, 0], axis=2)] = 5

    color[np.all(img==[150, 250, 0], axis=2)] = 6
    color[np.all(img==[150, 200, 150], axis=2)] = 7
    color[np.all(img==[200, 0, 200], axis=2)] = 8
    color[np.all(img==[150, 0, 250], axis=2)] = 9
    color[np.all(img==[150, 150, 250], axis=2)] = 10

    color[np.all(img==[250, 200, 0], axis=2)] = 11
    color[np.all(img==[200, 200, 0], axis=2)] = 12
    color[np.all(img==[0, 0, 200], axis=2)] = 13
    color[np.all(img==[0, 150, 200], axis=2)] = 14
    color[np.all(img==[0, 200, 250], axis=2)] = 15

    return color



def get_label_predict_top_k(predict_proprely, top_k):
    """
    image = load_image(image), input image is a ndarray
    return top-5 of label
    """
    # array 2 list
    # predict_proprely = model.predict(image)
    # print(np.argmax(predict_proprely)/100)
    predict_list = list(predict_proprely)
    min_label = min(predict_list)
    label_k = []
    for i in range(top_k):
        label = np.argmax(predict_list)
        #print(label)
        predict_list.remove(predict_list[label])
        predict_list.insert(label, min_label)
        label_k.append(label)
        # print(label_k)
    return label_k

def write_img(filename,im_proj,im_geotrans,im_data):
    #gdal数据类型包括
    #gdal.GDT_Byte,
    #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    #gdal.GDT_Float32, gdal.GDT_Float64

    #判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    #判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
    dataset.SetProjection(im_proj)                    #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

def generator_list_of_imagepath(path):
    image_list = []
    for image in os.listdir(path):
        # print(path)
        # print(image)
        if 'tif' == image.split('.')[-1]:
            # image_list.append(image.split('.')[0])
            image_list.append(path + '\\' + image)
    return image_list

def generator_list_of_labelpath(path):
    image_list = []
    for image in os.listdir(path):
        # print(path)
        # print(image)
        if 'npy' == image.split('.')[-1]:
            # image_list.append(image.split('.')[0])
            image_list.append(path + '\\' + image)
    return image_list

def main():
    # label_dict = {'Built-up': 0, 'bareland': 1, 'vegetation_coverage': 2, 'water': 3}
    # label_dict = {'arbor_woodland': 0, 'artificial_grassland': 1, 'dry_cropland': 2, 'garden_plot': 3, 'industrial_land': 4, 'irrigated_land': 5, 'lake': 6, 'natural_grassland': 7, 'paddy_field': 8, 'pond': 9, 'river': 10, 'rural_residential': 11, 'shrub_land': 12, 'traffic_land': 13, 'urban_residential': 14}
    label_dict = {'NotUV': 0, 'UV': 1}
    model = OS('.\\model-UV-Semantic\\UV-Semantic_CJZY-semiformer-0.5#1.pth')

    path_file_out = '.\\CJZY-UV-Confident-0.9-semiformer-0.5-out\\'

    if not os.path.exists(path_file_out):
        # remove_dir(converted_path)
        os.mkdir(path_file_out)

    block_id_list = []
    file = []

    img = 'M:\\UV-China\\China_Grid\\CJZY\\'  # 待预测图片文件夹

    imgs = generator_list_of_imagepath(img)
    # labels = generator_list_of_labelpath(label)
    # for idx in range(len(imgs)):
    #     print(imgs[idx], labels[idx])
    data_loader = open('.\CJZY_UV_Result.txt', 'r', encoding='utf-8')

    for line in data_loader:
        block_id = line.split('\t')[0]
        labels = line.split('\t')[1].replace('\n', '')
        # print(images_path, labels)
        # if int(labels) == 1:
            # print(images_path, labels)
    #         images = Image.open('M:\\UV-China\\China_Grid\\CJZY\\' + images_path + '.tif').convert('RGB')
    #
    # for img_path in imgs:
            # block_id = str(img_path.split('\\')[-1].split('.')[0])
            # img_path = '.\\data\\H_rs\\block' + str(blockid) + '.tif'
            # img = Image.open(img_path).convert('RGB')
        img_path = 'M:\\UV-China\\China_Grid\\CJZY\\' + str(block_id) + '.tif'
        dataset = gdal.Open(img_path)
        im_width = dataset.RasterXSize #栅格矩阵的列数
        im_height = dataset.RasterYSize #栅格矩阵的行数

        img = dataset.ReadAsArray(0 , 0,im_width,im_height).transpose(1, 2, 0)  # 2 3 4 5 8 11
        # img_6bands = np.zeros((555, 555, 6))
        # # 465  # 2 3 4 5 8 11
        # img_6bands[:, :, 0:4] = img[:, :, 2:6]
        # img_6bands[:, :, 4] = img[:, :, 7]
        # img_6bands[:, :, 5] = img[:, :, 10]
        # img = img_6bands
        # return im_data
        # dsm = np.array(cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED))
        # label = np.load(label_path).astype(int)
        # print(label.tolist())
        # data = Image.open(img_path).convert('RGB')
        # print(np.array(img).transpose((2,0,1)).shape, dsm.transpose((2,0,1)).shape)
        print('overlap滑动窗口切割')

        predict_list = predict_sliding(model, np.array(img), (465, 465), 2, 0)
        # print(predict_list, predict_list.shape)
        Confident_score = 0.9

        print('label to image')
        label_image = np.zeros((np.array(img).shape[0], np.array(img).shape[1]))
        # print(label_image.shape)
        label_image = predict_list[1, :, :] / (predict_list[0, :, :] + predict_list[1, :, :])
        # print(np.max(label_image))
        label_image[label_image<Confident_score] = 0
        label_image[label_image>=Confident_score] = 1
        # for i in range(0, predict_list.shape[1]):
        #     # if i % 1000 == 0:
        #     #     print('i:', i)
        #     for j in range(0, predict_list.shape[2]):
        #         # print(np.argmax(list(predict_list[:, i, j])))
        #         label_image[i][j] = list(predict_list[:, i, j])[np.argmax(list(predict_list[:, i, j]))] # get_label_predict_top_k(list(predict_list[:, i, j]), 1)[0]
        predict_image = color_predicts(label_image)
        # label_GT = color_predicts(label)
        # print(predict_image.shape)


        write_path = path_file_out + block_id + '_pred.png'
        write_path_oneband = path_file_out + block_id + '_label_pred_oneband.png'
        # write_path_label = path_file_out + block_id + '_label_GT.png'
        # print('write label image to file', write_path)
        # write_img(write_path, im_proj, im_geotrans, predict_image[:,:,:])
        # print(predict_image)
        cv2.imwrite(write_path, predict_image)
        # cv2.imwrite(write_path_label, label_GT)
        cv2.imwrite(write_path_oneband, label_image)


if __name__ == '__main__':
    main()
