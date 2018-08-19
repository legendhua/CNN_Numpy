# -*- coding: utf-8 -*-
'''
the implementation for conv2d with numpy
author: zhang guanghua
reference: https://blog.csdn.net/legend_hua/article/details/81590979
date: 2018-0814
'''
import numpy as np
import math
from functools import reduce

class Conv2D(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
        '''
        Arguments:
        shape----------the shape of input data 
        output_channels--------the output channel
        ksize------------------the kernel size
        stride-----------------the stride size
        method-----------------the edge processing method in convolution
        '''
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = self.input_shape[-1]
        self.batchsize = self.input_shape[0]
        self.stride = stride
        self.ksize = ksize
        self.method = method
        
        # weight initialization
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, self.input_shape) / self.output_channels)
        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        # bias initialization
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        
        # the error in back propagation
        if method == 'VALID':
            self.err = np.zeros((self.input_shape[0], math.ceil((self.input_shape[1] - ksize + 1) / self.stride), math.ceil((self.input_shape[1] - ksize + 1) / self.stride),
             self.output_channels))

        if method == 'SAME':
            self.err = np.zeros((self.input_shape[0], math.ceil(self.input_shape[1]/self.stride), math.ceil(self.input_shape[2]/self.stride),self.output_channels))
        
        # gradient initialization
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.err.shape


    def forward(self, input):
        self.input = input
        #print(self.weights)
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            self.input = np.pad(self.input, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                             'constant', constant_values=0)
        
        self.col_image = []
        conv_out = np.zeros(self.output_shape)
        for i in range(self.batchsize):
            img_i = self.input[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            #print(self.col_image_i.shape)
            #print(col_weights.shape)
            a = np.dot(self.col_image_i, col_weights) + self.bias
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.err[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out
    
    # computing the w_grad and b_grad, & computing the next err
    def gradient(self, err):
        self.err = err
        col_err = np.reshape(err, [self.batchsize, -1, self.output_channels])
        
        # computing the w_gre and b_gre by input and output error 
        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_err[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_err, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_err(front layer)
        # padding the 
        if self.method == 'VALID':
            pad_err = np.pad(self.err, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_err = np.pad(self.err, (
                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
                             'constant', constant_values=0)
        
        # Flip the weights in the up/dowm and right/left direction
        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_err = np.array([im2col(pad_err[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_err = np.dot(col_pad_err, col_flip_weights)
        next_err = np.reshape(next_err, self.input_shape)
        return next_err

    def backward(self, err, alpha=0.00001, weight_decay=0.0004):
        next_err = self.gradient(err)
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.b_gradient

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        
        return next_err


def im2col(image, ksize, stride):
    # the shape of image is [batchsize, width ,height, channel]
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col


if __name__ == "__main__":
    # img = np.random.standard_normal((2, 32, 32, 3))
    img = np.ones((1, 28, 28, 3))
    conv = Conv2D([1,28,28,3], 12, 3, 1)
    output = conv.forward(img)
    err1 = output.copy() + 0.1 - output
    conv.backward(err1)
    print(conv.w_gradient)
    print(conv.b_gradient)