# -*- coding: utf-8 -*-
'''
Some helper functions
author: zhang guanghua
date: 2018-0814
'''
import numpy as np
import glob
import os
import sys
import gzip
import struct
import cv2
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from six.moves import urllib

LOG_DIR = os.path.join(BASE_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

MNIST_DATA_PATH = r'F:\my_learning\DL\tensorflow\mnist\mnist_data'


 
def maybe_download(filename, data_dir, SOURCE_URL):
    """Download the data from Yann's website, unless it's already here."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
 
 
def check_file(data_dir):
    if os.path.exists(data_dir):
        return True
    else:
        os.mkdir(data_dir)
        return False


def _read_data(gzfname):
    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        # 读取magic number
        if n[0] != 0x3080000:
            raise Exception('文件无效: 未预期的magic number.')
        # 将数据整体读入
        n = struct.unpack('>I', gz.read(4))[0]
        crow = struct.unpack('>I', gz.read(4))[0]
        ccol = struct.unpack('>I', gz.read(4))[0]
        if crow != 28 or ccol != 28:
            raise Exception('文件无效: 每张图像应该有 28 行/列.')
        # 读取数据
        res = np.fromstring(gz.read(n * crow * ccol), dtype = np.uint8)
        
    return res.reshape((n, crow, ccol, 1))

def _read_label(gzfname):
    with gzip.open(gzfname) as gz:
        n = struct.unpack('I', gz.read(4))
        # 读取magic number
        if n[0] != 0x1080000:
            raise Exception('文件无效: 未预期的 magic number.')
        # 将数据整体读入
        n = struct.unpack('>I', gz.read(4))[0]
        
        # 读取标签
        res = np.fromstring(gz.read(n), dtype = np.uint8)
        #n_class = res.max() + 1
        #res_one_hot = np.eye(n_class)[res]
    return res.reshape((n,1))

def load_data(data_dir, isTraing=True):
    if check_file(data_dir):
        print(data_dir)
        print('dir mnist already exist.')
    
    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    data_keys = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    
    train_imgs = []
    val_imgs = []
    test_imgs = []
    train_label = []
    val_label = []
    test_label = []

    if isTraing:
        # image_data
        if os.path.isfile(os.path.join(data_dir, data_keys[0])):
             print("[warning...]", data_keys[0], "already exist.")
        else:
            maybe_download(data_keys[0], data_dir, SOURCE_URL)
        image_data = _read_data(os.path.join(data_dir, data_keys[0]))
        #print(image_data.shape)
        train_imgs = image_data[0:55000,:]
        val_imgs = image_data[55000:,:]

        # label_data
        if os.path.isfile(os.path.join(data_dir, data_keys[1])):
             print("[warning...]", data_keys[1], "already exist.")
        else:
            maybe_download(data_keys[1], data_dir, SOURCE_URL)
        label_data = _read_label(os.path.join(data_dir, data_keys[1]))
        #print(label_data)
        train_label = label_data[0:55000,:]
        val_label = label_data[55000:,:]
        
    else:
        # image_data
        if os.path.isfile(os.path.join(data_dir, data_keys[2])):
             print("[warning...]", data_keys[2], "already exist.")
        else:
            maybe_download(data_keys[2], data_dir, SOURCE_URL)
        test_imgs = _read_data(os.path.join(data_dir, data_keys[2]))
        #print(test_imgs.shape)
        # label_data
        if os.path.isfile(os.path.join(data_dir, data_keys[3])):
             print("[warning...]", data_keys[3], "already exist.")
        else:
            maybe_download(data_keys[3], data_dir, SOURCE_URL)
        test_label = _read_label(os.path.join(data_dir, data_keys[3]))
        #print(test_label.shape)
        
    return train_imgs, val_imgs, test_imgs, train_label, val_label, test_label

def mnist2img():
    image_path = os.path.join(BASE_DIR,'image_path')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
        
    train_imgs,_,_,_,_,_ = load_data(MNIST_DATA_PATH)
    for i in range(1000):
        print(i)
        img_name = os.path.join(image_path, 'mnist_image{}.png'.format(i))
        cv2.imwrite(img_name, train_imgs[i])

def draw_loss_curve():
    with open(os.path.join('log', 'log_train.txt')) as file:
        train_loss = []
        train_acc = []
        logs = file.readlines()
        for log in logs[6::4]:
            key = log.split(':')[0]
            if 'train mean loss' == key:
                value = log.split(':')[1]
                train_loss.append(float(value))
        for log in logs[7::4]:
            key = log.split(':')[0]
            if 'train accuracy' == key:
                value = log.split(':')[1]
                train_acc.append(float(value))

    x = range(len(train_loss))
    plt.plot(x,train_loss,color='blue',label='train loss')
    plt.plot(x,train_acc,color='orange',label='train accuracy')
    plt.title('Mnist detect train loss & accuracy')
    plt.legend()
    plt.show()
    
        
if __name__ == '__main__':
    #load_data(MNIST_DATA_PATH)
    #load_data()
    #mnist2img()
    draw_loss_curve()
    