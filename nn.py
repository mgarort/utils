# This module will contain utilities related to neural networks

import numpy as np
from collections.abc import Iterable


def compute_dims_after_conv3d(x, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    """This utility function calculates the output dimensions after a 3D convolution.
    - args: It takes exactly the same parameters as the conv3d layer, so that we can pass exactly the same parameter dictionary. This is
            convenient, but note that some parameters (like bias=True) will never be used.
    returns: a tuple with the dimensions that the output would have, i.e. a tuple (n_out, c_out, d_out, h_out, w_out), where:
                    - n_out: batch size (same as n_in, always for every layer)
                    - c_out: number of channels (specified by the user in the conv3d layer)
                    - d_out: depth, depends on the kernel size, stride, padding and dilation
                    - h_out: height, idem
                    - w_out: width, idem
    """

    # Variables to use in computation
    n_in, c_in, d_in, h_in, w_in = x.shape
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
    n_out = n_in  # The batch size is always maintained
    c_out = out_channels
    d_out, h_out, w_out = np.floor( (np.array([d_in, h_in, w_in]) + 2*padding - dilation*(kernel_size-1) - 1) /stride + 1)

    return (int(n_out), int(c_out), int(d_out), int(h_out), int(w_out))


def compute_dims_after_maxpool3d(x, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    """This utility function calculates the output dimensions after a 3D max pool operation.
    - args: It takes exactly the same parameters as the maxpool3d layer, so that we can pass exactly the same parameter dictionary. This is
            convenient, but note that some parameters (like bias=True) will never be used.
    returns: a tuple with the dimensions that the output would have, i.e. a tuple (n_out, c_out, d_out, h_out, w_out), where:
                    - n_out: batch size (same as n_in, always for every layer)
                    - c_out: number of channels (same as c_in for the pooling layer)
                    - d_out: depth, depends on the kernel size, stride, padding and dilation
                    - h_out: height, idem
                    - w_out: width, idem
    """

    # Variables to use in computation
    n_in, c_in, d_in, h_in, w_in = x.shape
    # By default, if stride is not specified, it is set to be the same as the kernel size
    if stride is None:
        stride = kernel_size
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
    n_out = n_in  # The batch size is always kept the same
    c_out = c_in  # In a pooling operation, the number of channels doesn't change
    round_mode = np.ceil if ceil_mode else np.floor # Strangely, the maxpool3d layer accepts ceiling instead of flooring
    d_out, h_out, w_out = round_mode( (np.array([d_in, h_in, w_in]) + 2*padding - dilation*(kernel_size-1) - 1) /stride + 1)

    return (int(n_out), int(c_out), int(d_out), int(h_out), int(w_out))
