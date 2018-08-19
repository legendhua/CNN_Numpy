# -*- coding: utf-8 -*-
'''
test process
author: zhang guanghua
date: 2018-0814
'''

import sys
import os
import argparse
import pickle
import util
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from lenet import LeNet
import numpy as np

BATCH_SIZE = 4

LOG_DIR = os.path.join(BASE_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = open(os.path.join(LOG_DIR,'log_test.txt'), 'w')

parser = argparse.ArgumentParser(description='Mnist Detect')
parser.add_argument('--path', '-p', default=os.path.join(BASE_DIR,'mnist_data'),metavar='Mnist Directory',
                    help='Please Input Mnist Path!')
parser.add_argument('--model', '-m', default=os.path.join(BASE_DIR,'model', 'model.pkl'),metavar='Model Directory',
                    help='Please Input Model Path!')
#MODEL_FILE = os.path.join(BASE_DIR, 'model', 'model99.pkl')

args = parser.parse_args()
MNIST_PATH = args.path
MODEL_FILE = args.model

#MNIST_PATH = r'F:\my_learning\DL\tensorflow\mnist\mnist_data'

def log_string(out_str):
    LOG_FILE.write(out_str+'\n')
    LOG_FILE.flush()
    print(out_str)
    


def LeNet_test():
    # initializing the network
    network = LeNet(BATCH_SIZE)
    network.getTheParas(MODEL_FILE)
    
    # load the test data
    _, _, test_imgs, _, _, test_label = util.load_data(MNIST_PATH, False)
    
    log_string('------------start test-----------')
    
    num_batch = test_imgs.shape[0] // BATCH_SIZE
    start = 0
    end = start + BATCH_SIZE
    loss = 0.0
    total_correct = 0.0
    total_seen = 0
    for n in range(num_batch):
        log_string('--------{}/{}(batchs) completed!'.format(n+1,num_batch))
        current_img = test_imgs[start:end,...]
        current_label = test_label[start:end,...]
        start = end
        end += BATCH_SIZE
        predict_val, loss_val = network.forward(current_img, current_label)
        correct = np.sum(predict_val == current_label)
        total_correct += correct
        loss += loss_val
        total_seen += BATCH_SIZE 
    log_string('eval mean loss: {}'.format(loss/num_batch))
    log_string('eval accuracy: {}'.format(total_correct/total_seen))

if __name__ == '__main__':
    import time
    start = time.time()
    LeNet_test()
    print('Time in total:', time.time()-start)
