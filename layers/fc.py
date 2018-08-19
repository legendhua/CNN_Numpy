# -*- coding: utf-8 -*-
'''
the implementation for pooling with numpy
author: zhang guanghua
reference: https://blog.csdn.net/legend_hua/article/details/81584880
date: 2018-0813
'''

import numpy as np


class FullyConnect(object):
    def __init__(self, shape, output_num=2):
        '''
        Arguments:
        shape----------the shape of input data(2 Dims:[batch, n])
        output_num-----------the length of output vector
        '''
        
        self.input_shape = shape
        self.batchsize = shape[0]
        input_len = self.input_shape[1]
        self.weights = np.random.standard_normal((input_len, output_num)) / 200
        self.bias = np.random.standard_normal(output_num) / 200

        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self,input):
        self.input = input
        output = np.dot(self.input, self.weights)+self.bias
        return output

    def gradient(self, err):
        for i in range(err.shape[0]):
            col_input = self.input[i][:, np.newaxis]
            err_i = err[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_input, err_i)
            self.b_gradient += err_i.reshape(self.bias.shape)

        next_err = np.dot(err, self.weights.T)
        next_err = np.reshape(next_err, self.input_shape)

        return next_err

    def backward(self, err, alpha=0.00001, weight_decay=0.0004):
        next_err = self.gradient(err)
        
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.b_gradient
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        
        return next_err

if __name__ == "__main__":
    img = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    fc = FullyConnect([2,6], 2)
    out = fc.forward(img)

    fc.backward(np.array([[1, -2],[3,4]]))

    print(fc.w_gradient)
    print(fc.b_gradient)

    
    print(fc.weights)
