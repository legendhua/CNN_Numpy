# -*- coding: utf-8 -*-
'''
the implementation for softmax with numpy
author: zhang guanghua
date: 2018-0814
'''

import numpy as np

class Softmax(object):
    def __init__(self, shape):
        self.err = np.zeros(shape)
        self.batchsize = shape[0]
        self.softmax = np.zeros(shape)
        
    def predict(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        for i in range(self.batchsize):
            prediction[i, :] -= np.max(prediction[i, :]) # more stable
            exp_prediction[i,:] = np.exp(prediction[i,:], dtype = np.float32)
            self.softmax[i,:] = exp_prediction[i,:]/np.sum(exp_prediction[i,:])
        return self.softmax

    def cal_loss(self, prediction, label):
        self.predict(prediction)
        self.label = label
        self.loss = 0
        for i in range(self.batchsize):
            self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]
        return (self.loss/self.batchsize)[0]


    # d_J/d_z
    def backward(self):
        self.err = self.softmax.copy()
        for i in range(self.batchsize):
            #print(np.where(self.label[i]==1))
            self.err[i, self.label[i]] -= 1
        return self.err
    
if __name__ == '__main__':
    predict = np.array([[0.01, 0.05, 0.04, 0.8, 0.6, 0.04], [0.9, 0, 0.03, 0, 0.05, 0.02]])
    label = np.array([[3], [0]])
    softmax = Softmax([2,6])
    print(softmax.predict(predict))
    print(softmax.cal_loss(predict,label))
    print(softmax.backward())
    
