import numpy as np
import pickle
import cv2
from PIL import Image
import os
def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
        return X
#X = load_CIFAR_batch('/home/scw4750/LJQ/MDCN/dataset/DIV2K/DIV2K_train_LR_bicubic/X1')   
#X = load_CIFAR_batch('/home/scw4750/LJQ/MDCN/dataset/cifar-10/cifar-10-batches-py/test_batch')
#img = X
path  = os.getcwd() + '/dataset/DIV2K/DIV2K_train_LR_bicubic/X1'

def read_image(path):
    image_array = []
    file_list = os.listdir(path)
    print(len(file_list))
    for file in file_list:
        image = cv2.imread(path + '/' + file)  # 打开图片
        file_name = file.split(".")[0]
        image_array.append(image)
    return image_array

img = read_image(path)
#img = cv2.imread(path)
R_means = []
G_means = []
B_means = []
R_stds = []
G_stds = []
B_stds = []
for im in img:
    im_R = im[:,:,0]/255
    im_G = im[:,:,1]/255
    im_B = im[:,:,2]/255
    im_R_mean = np.mean(im_R)
    im_G_mean = np.mean(im_G)
    im_B_mean = np.mean(im_B)
    im_R_std = np.std(im_R)
    im_G_std = np.std(im_G)
    im_B_std = np.std(im_B)
    R_means.append(im_R_mean)
    G_means.append(im_G_mean)
    B_means.append(im_B_mean)
    R_stds.append(im_R_std)
    G_stds.append(im_G_std)
    B_stds.append(im_B_std)
a = [R_means,G_means,B_means]
b = [R_stds,G_stds,B_stds]
mean = [0,0,0]
std = [0,0,0]
mean[0] = np.mean(a[0])
mean[1] = np.mean(a[1])
mean[2] = np.mean(a[2])
std[0] = np.mean(b[0])
std[1] = np.mean(b[1])
std[2] = np.mean(b[2])
print('数据集的RGB平均值为\n[{},{},{}]'.format(mean[0],mean[1],mean[2]))
print('数据集的RGB方差为\n[{},{},{}]'.format(std[0],std[1],std[2]))