# -*- coding: utf-8 -*-
'''
the implementation for pooling with numpy
author: zhang guanghua
reference: https://blog.csdn.net/legend_hua/article/details/81590979
date: 2018-0813
'''

import numpy as np


class MaxPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        '''
        Arguments:
        shape----------the shape of input data
        ksize------------------the kernel size
        stride-----------------the stride size
        '''
        
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = self.input_shape[-1]
        self.index = np.zeros(self.input_shape)
        self.output_shape = [self.input_shape[0], self.input_shape[1] // self.stride, self.input_shape[2] // self.stride, self.output_channels]
        
        self.output_shape_resize = [self.input_shape[0], (self.input_shape[1] // self.stride) *(self.input_shape[2] // self.stride)*(self.output_channels)]
        
    def forward(self, input):
        self.input = input
        out = np.zeros([self.input.shape[0], self.input.shape[1] // self.stride, self.input.shape[2] // self.stride, self.output_channels])

        for b in range(self.input.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, self.input.shape[1], self.stride):
                    for j in range(0, self.input.shape[2], self.stride):
                        out[b, i // self.stride, j // self.stride, c] = np.max(
                            self.input[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(self.input[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i+index//self.stride, j + index % self.stride, c] = 1
        return out
    
    # upsample the err to get next err(front layer)
    def backward(self, err):
        err = np.reshape(err, self.output_shape)
        return np.repeat(np.repeat(err, self.stride, axis=1), self.stride, axis=2) * self.index
    

if __name__ == "__main__":
    img = np.array([[1,2,3,4],[3,4,1,2],[4,5,6,7],[6,7,4,5]])
    img = img[np.newaxis, :]
    img = np.concatenate((img,img))
    img = img[:, :, :,np.newaxis]
    maxPool = MaxPooling([2,4,4,1])
    output = maxPool.forward(img)
    next = output.copy() + 2
    next1 = maxPool.backward(next - output)
    print(next1)
    #print(maxPool.index)