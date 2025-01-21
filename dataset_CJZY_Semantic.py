import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as io
import torchvision
from osgeo import gdal
from torchvision import transforms as T
import numpy as np
import cv2
# gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
def MyLoader(path,type):
    if type=='img':
        return Image.open(path).convert('RGB')
    elif type=='tif':
        # print(path)
        dataset = gdal.Open(path)
        im_width = dataset.RasterXSize #栅格矩阵的列数
        im_height = dataset.RasterYSize #栅格矩阵的行数

        im_data = dataset.ReadAsArray(0,0,im_width,im_height).transpose(1, 2, 0)  # 2 3 4 5 8 11
        return im_data
    # elif type=='label':
    #     return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif type == 'label':
        return np.array(Image.open(path)).astype(int)

# img = MyLoader("M:\\长江中游城市群开放空间语义分割数据集\\长江中游\\AOI_2_raster_02degrees\\466.tif",'label')
# print(img)


def bgr2gray(img):
    color = np.zeros([img.shape[0], img.shape[1]])  # BGR
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         # if img[i, j, :] ==
    #         #     print(img[i, j, :])
    #         if (img[i, j, :] == np.array([0, 0, 0])).all():
    #             color[i, j] = 0
    #         if (img[i, j, :] == np.array([0, 128, 0])).all():
    #             color[i, j] = 1
    #         if (img[i, j, :] == np.array([0, 128, 128])).all():
    #             color[i, j] = 2
    #         if (img[i, j, :] == np.array([0, 0, 128])).all():
    #             color[i, j] = 3
    # print(np.all(img==[0, 0, 0], axis=2))
    # print(np.all(img==[0, 128, 0], axis=2))
    color[np.all(img==[0, 0, 0], axis=2)] = 0 # 黑色 其他
    color[np.all(img==[0, 128, 0], axis=2)] = 1 # 绿色 户外停车场
    # color[np.all(img==[0, 0, 128], axis=2)] = 2  # 红色 建筑物
    # color[np.all(img==[0, 128, 128], axis=2)] = 3 # 黄色 户外运动场 # 黄色
    # # np.savetxt('color.txt', color, fmt="%i")
    # # imsave('.\\Label_to_Tif\\002005_label_mosaic_oneband.png', color)
    # cv2.imwrite('.\\Label_to_Tif\\002005_label_mosaic_oneband.png', color)
    # imsave(img_add_path, img_add)
    return color

class UVdataset(Dataset):
    def __init__(self,txt,transform=None, target_transform=None, loader=MyLoader):
        file = []
        with open(txt,'r', encoding='utf-8') as fh:
            for line in fh:
                # count = 0
                # print(line)
                line=line.strip('\n')
                line=line.rstrip()
                words=line.split()
                # print(words)
                img_path = str(words[0])
                label_path = str(words[1])
                # print(imapath, labelpath)
                file.append((img_path, label_path))


        self.file=file
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader


    def __getitem__(self,index):

        img, label = self.file[index]
        # print(img, label)
        # img_6bands = np.zeros((555, 555, 6))
        img = self.loader(img,type='tif')[:465, :465, :]  # 465  # 2 3 4 5 8 11
        # img_6bands[:, :, 0:4] = img[:, :, 2:6]
        # img_6bands[:, :, 4] = img[:, :, 7]
        # img_6bands[:, :, 5] = img[:, :, 10]

        label = self.loader(label,type='label')[:465, :465]

        # label[label==0] = 2
        # label = label-1
        # label[label==3] = 1
        # # label[label==0] = 1
        # # label = label - 1
        # # label[label==3] = 2
        # label[label==4] = 3
        label[label==2] = 0
        # label[label==6] = 5
        # print(img.shape, label.shape)

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.file)

class UVSemidataset(Dataset):
    def __init__(self,txt,transform=None, target_transform=None, loader=MyLoader):
        file = []
        with open(txt,'r', encoding='utf-8') as fh:
            for line in fh:
                # count = 0
                # print(line)
                line=line.strip('\n')
                line=line.rstrip()
                words=line.split()
                # print(words)
                img_path = str(words[0])
                label_path = str(words[1])
                # print(imapath, labelpath)
                file.append((img_path, label_path))


        self.file=file
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader


    def __getitem__(self,index):

        img, label = self.file[index]
        # print(img, label)
        # img_6bands = np.zeros((555, 555, 6))
        img = self.loader(img,type='tif')[:465, :465, :]  # 465  # 2 3 4 5 8 11
        # img_6bands[:, :, 0:4] = img[:, :, 2:6]
        # img_6bands[:, :, 4] = img[:, :, 7]
        # img_6bands[:, :, 5] = img[:, :, 10]

        label = self.loader(label,type='label')[:465, :465]

        # label[label==0] = 3
        # # label = label-1
        # label[label==2] = 0
        # label[label==3] = 2

        # # label = label - 1
        # # label[label==3] = 2
        # label[label==4] = 3
        # label[label==2] = 0
        # label[label==6] = 5
        # print(img.shape, label.shape)

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.file)

if __name__ == "__main__":
    test_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_dataset=UVSemidataset(txt='.\\data\\trainUV_Semantic_CJZY.txt',transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,pin_memory=True)
    for step,(img, label) in enumerate(test_loader):
        print(img.shape, label[:, :, 300, 300])
