# -*- coding: utf-8 -*-
'''
train process
author: zhang guanghua
date: 2018-0814
'''

import sys
import os
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import util
import time
from lenet import LeNet
import numpy as np

BATCH_SIZE = 32
MAX_EPOCH = 20

LOG_DIR = os.path.join(BASE_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = open(os.path.join(LOG_DIR,'log_train.txt'), 'w')

parser = argparse.ArgumentParser(description='Mnist Detect')
parser.add_argument('--path', '-p', default=os.path.join(BASE_DIR,'mnist_data'),metavar='Mnist Directory',
                    help='Please Input Mnist Path!')
args = parser.parse_args()
MNIST_PATH = args.path

#MNIST_PATH = r'F:\my_learning\DL\tensorflow\mnist\mnist_data'

def log_string(out_str):
    LOG_FILE.write(out_str+'\n')
    LOG_FILE.flush()
    print(out_str)

def LeNet_train():
    # initializing the network
    network = LeNet(BATCH_SIZE)
    # load the data
    train_imgs, val_imgs, _, train_label, val_label, _ = util.load_data(MNIST_PATH)

    for epoch in range(MAX_EPOCH):
        eval_one_epoch(network, val_imgs, val_label)
        log_string('------------start train {}/{}----------'.format(epoch, MAX_EPOCH))
        train_one_epoch(network, train_imgs, train_label, epoch)
        
    
def train_one_epoch(net, img, label, epoch):
    num_batch = img.shape[0] // BATCH_SIZE
    start = 0
    end = start + BATCH_SIZE
    loss = 0.0
    total_correct = 0.0
    total_seen = 0
    #loss_train = []
    #loss_eval = []
    for n in range(num_batch):
        current_img = img[start:end,...]
        current_label = label[start:end,...]
        start = end
        end += BATCH_SIZE
        predict_val, loss_val = net.forward(current_img, current_label)
        correct = np.sum(predict_val == current_label)
        total_correct += correct
        loss += loss_val*BATCH_SIZE
        total_seen += BATCH_SIZE
        if (n+1)%100 == 0:
            log_string('-----------train loss & acc-----------')
            log_string('{}/{} (batchs) completed!'.format(n+1, num_batch))
            log_string('train mean loss: {}'.format(loss/total_seen))
            log_string('train accuracy: {}'.format(total_correct/total_seen))
            #loss_train.append(loss/total_seen)
        net.backward(epoch, MAX_EPOCH) 
    # save the model
    if (epoch+1)%1 == 0:
        net.saveTheModel(epoch)
        
def eval_one_epoch(net, img, label):
    log_string('------------start eval----------')
    num_batch = img.shape[0] // BATCH_SIZE
    start = 0
    end = start + BATCH_SIZE
    loss = 0.0
    total_correct = 0.0
    total_seen = 0
    
    for n in range(num_batch):
        current_img = img[start:end,...]
        current_label = label[start:end,...]
        start = end
        end += BATCH_SIZE
        predict_val, loss_val = net.forward(current_img, current_label)
        correct = np.sum(predict_val == current_label)
        total_correct += correct
        loss += loss_val
        total_seen += BATCH_SIZE 
    
    log_string('eval mean loss: {}'.format(loss/num_batch))
    log_string('eval accuracy: {}'.format(total_correct/total_seen))

        
if __name__ == '__main__':
    LeNet_train()
    LOG_FILE.close()        
    