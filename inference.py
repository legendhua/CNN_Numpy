# -*- coding: utf-8 -*-
'''
inference process
author: zhang guanghua
date: 2018-0814
'''

import sys
import os
import glob
import argparse
import pickle
import util
from numpy import newaxis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import cv2
from lenet import LeNet
import numpy as np

#MODEL_FILE = os.path.join(BASE_DIR, 'model', 'model99.pkl')
BATCH_SIZE = 1

parser = argparse.ArgumentParser(description='Mnist Detect')
parser.add_argument('--image_path', '-p', default=os.path.join(BASE_DIR,'image_path'),metavar='image Directory',
                    help='Please Input Image Path!')
parser.add_argument('--model', '-m', default=os.path.join(BASE_DIR,'model', 'model.pkl'),metavar='Model Directory',
                    help='Please Input Model Path!')
args = parser.parse_args()
IMAGE_PATH = args.image_path
MODEL_FILE = args.model

#IMAGE_PATH = r'F:\my_learning\algorithm_file\Python\libOrAlgorithm\mnistDetect\image_path'
    

def inference():
    # initializing the network
    network = LeNet(BATCH_SIZE)
    network.getTheParas(MODEL_FILE)
    print(IMAGE_PATH) 
    image_paths = glob.glob(os.path.join(IMAGE_PATH,'*'))
    
    for image_path in image_paths:
        image_data = cv2.imread(image_path, 0)
        image_data = image_data[newaxis,:,:,newaxis]
        predict_val = network.inference(image_data)
        print(image_path,':', predict_val[0][0])


if __name__ == "__main__":
    import time
    start = time.time()
    inference()
    print('Time in total:', time.time()-start)
