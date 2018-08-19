# -*- coding: utf-8 -*-
'''
the implementation for lenet 
author: zhang guanghua
date: 2018-0814
'''
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'layers'))
from conv2d import Conv2D
from pooling import MaxPooling
from fc import FullyConnect
from relu import Relu
from softmax import Softmax
import util
import pickle
import numpy as np

MODEL_PATH = os.path.join(BASE_DIR, 'model')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

class LeNet(object):
    def __init__(self, batchSize):
        # network structure 
        self.batchSize = batchSize
        self.conv1 = Conv2D([self.batchSize,28,28,1], 6, 5)

        self.relu1 = Relu(self.conv1.output_shape)
        
        self.pool1 = MaxPooling(self.relu1.output_shape)
        
        self.conv2 = Conv2D(self.pool1.output_shape, 16, 5)
        
        self.relu2 = Relu(self.conv2.output_shape)
        
        self.pool2 = MaxPooling(self.relu2.output_shape)
        
        self.fc1 = FullyConnect(self.pool2.output_shape_resize, 120)
        
        self.relu3 = Relu(self.fc1.output_shape)
        
        self.fc2 = FullyConnect(self.relu3.output_shape, 84)
        
        self.relu4 = Relu(self.fc2.output_shape)
        
        self.fc3 =FullyConnect(self.relu4.output_shape, 10)
        
        self.soft = Softmax(self.fc3.output_shape)
    
    def inference(self, input):
        # inference
        self.input = input
        conv1_out = self.conv1.forward(input)
        relu1_out = self.relu1.forward(conv1_out)
        pool1_out = self.pool1.forward(relu1_out)
        conv2_out = self.conv2.forward(pool1_out)
        relu2_out = self.relu2.forward(conv2_out) 
        pool2_out = self.pool2.forward(relu2_out)
        pool2_out = pool2_out.reshape([self.batchSize,-1])
        fc1_out = self.fc1.forward(pool2_out)
        relu3_out = self.relu3.forward(fc1_out)
        fc2_out = self.fc2.forward(relu3_out)
        relu4_out = self.relu4.forward(fc2_out)
        fc3_out = self.fc3.forward(relu4_out)
        predict = self.soft.predict(fc3_out)
        predict = np.argmax(predict, 1).reshape([self.batchSize, 1])
        return predict
    
    def forward(self, input, label):
        # forward
        self.input = input
        self.label = label
        conv1_out = self.conv1.forward(input)
        relu1_out = self.relu1.forward(conv1_out)
        pool1_out = self.pool1.forward(relu1_out)
        conv2_out = self.conv2.forward(pool1_out)
        relu2_out = self.relu2.forward(conv2_out) 
        pool2_out = self.pool2.forward(relu2_out)
        pool2_out = pool2_out.reshape([self.batchSize,-1])
        fc1_out = self.fc1.forward(pool2_out)
        relu3_out = self.relu3.forward(fc1_out)
        fc2_out = self.fc2.forward(relu3_out)
        relu4_out = self.relu4.forward(fc2_out)
        fc3_out = self.fc3.forward(relu4_out)
        predict = self.soft.predict(fc3_out)
        loss = self.soft.cal_loss(fc3_out, self.label)
        predict = np.argmax(predict, 1).reshape([self.batchSize, 1])
        return predict, loss   

    def backward(self, epoch, maxEpoch):
        learning_rate = [0.001, 0.0001]
        if epoch < maxEpoch//3:
            current_learning_rate = learning_rate[0]
        else:
            current_learning_rate = learning_rate[1]
        self.conv1.backward(alpha=current_learning_rate, err=self.relu1.backward(self.pool1.backward(self.conv2.backward(alpha=current_learning_rate, err=self.relu2.backward(self.pool2.backward(
            self.fc1.backward(alpha=current_learning_rate, err=self.relu3.backward(self.fc2.backward(alpha=current_learning_rate, err=self.relu4.backward(self.fc3.backward(alpha=current_learning_rate, err=self.soft.backward())))))))))))
    
    def _saveParasToJson(self, epoch):
        name_dict = {'conv1':self.conv1,'conv2':self.conv2,'fc1':self.fc1,'fc2':self.fc2,'fc3':self.fc3}
        model = {'conv1':{},'conv2':{},'fc1':{},'fc2':{},'fc3':{}}
        for layer in model:
            if 'weights' not in model[layer]:
                model[layer]['weights'] = name_dict[layer].weights
            if 'bias' not in model[layer]:
                model[layer]['bias'] = name_dict[layer].bias
        modelname = open(os.path.join(MODEL_PATH,'model{}.pkl'.format(epoch)),'wb')
        pickle.dump(model, modelname, -1)
        modelname.close()
        
    def saveTheModel(self, epoch):
        if (epoch+1)%1 == 0:
            self._saveParasToJson(epoch)
            
    def getTheParas(self, model_path):
        f = open(model_path, 'rb')
        model_dict = pickle.load(f)
        self.conv1.weights = np.array(model_dict['conv1']['weights'])
        self.conv1.bias = np.array(model_dict['conv1']['bias'])
        self.conv2.weights = np.array(model_dict['conv2']['weights'])
        self.conv2.bias = np.array(model_dict['conv2']['bias'])
        self.fc1.weights = np.array(model_dict['fc1']['weights'])
        self.fc1.bias = np.array(model_dict['fc1']['bias'])
        self.fc2.weights = np.array(model_dict['fc2']['weights'])
        self.fc2.bias = np.array(model_dict['fc2']['bias'])
        self.fc3.weights = np.array(model_dict['fc3']['weights'])
        self.fc3.bias = np.array(model_dict['fc3']['bias']) 
        f.close()    
        
if __name__ == '__main__':
    train_imgs, val_imgs, test_imgs, train_label, val_label, test_label = util.load_data(r'F:\my_learning\DL\tensorflow\mnist\mnist_data')
    #img = np.ones((2,28,28,1)) 
    net = LeNet(2) 
    for i in range(100):
        print(i)
        net.forward(train_imgs[0:2,...], train_label[0:2,...])
        net.backward(i, 100) 
        net.saveTheModel(i)
    print('predict:',net.forward(train_imgs[0:2,...], train_label[0:2,...]))
    print('truth', train_label[0:2,...])
    