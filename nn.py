# This module will contain utilities related to neural networks

import numpy as np


def compute_dims_after_conv3d(x, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    """This utility function calculates the output dimensions after a 3D convolution.
    - args: It takes exactly the same parameters as the conv3d layer, so that we can pass exactly the same parameter dictionary. This is
            convenient, but note that some parameters (like bias=True) will never be used.
    returns: a dictionary with the dimensions n (batch size, c_out (channels out, this is specified by the user in the definition of the 
             3D convolutional layer), d_out (depth out), h_out (height out), w_out (width out)
    """

    # Variables to use in computation
    n, c_in, d_in, h_in, w_in = x.shape
    assert c_in == in_channels, 'Number of input channels in the image is not the same as the number of input channels specified'
    # If kernel size, stride, padding and dilation are not given as iterables, make them tuples
    if not isinstance(kernel_size, Iterable):
        kernel_size = np.array([kernel_size,]*3)
    if not isinstance(stride, Iterable):
        stride = np.array([stride,]*3)
    if not isinstance(padding, Iterable):
        padding = np.array([padding,]*3)
    if not isinstance(dilation, Iterable):
        dilation = np.array([dilation,]*3)

    # Variables to return
    c_out = out_channels
    d_out, h_out, w_out = np.floor( (np.array([d_in, h_in, w_in]) + 2*padding - dilation*(kernel_size-1) - 1) /stride + 1)

    return {'n': int(n), 'c_out': int(c_out), 'd_out':int(d_out), 'h_out':int(h_out), 'w_out':int(w_out)}
