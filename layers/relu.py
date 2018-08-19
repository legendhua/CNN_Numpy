# -*- coding: utf-8 -*-
'''
the implementation for pooling with numpy
author: zhang guanghua
date: 2018-0813
'''

import numpy as np

class Relu(object):
    def __init__(self, shape):
        self.input_shape = shape
        self.err = np.zeros(self.input_shape)
        self.output_shape = self.input_shape

    def forward(self, input):
        self.input = input
        return np.maximum(self.input, 0)

    def backward(self, err):
        self.err = err
        self.err[self.input<0]=0
        return self.err
    


if __name__ == '__main__':
    img = np.array([[1, 2, 3, 4, 5, 6], [-6, -5, -4, -3, -2, -1]])
    relu = Relu([2,6])
    output = relu.forward(img)
    print(output)
    err1 = output.copy() + 0.1 - output
    print(relu.backward(err1))