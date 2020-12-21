# This module will contain utilities related to neural networks

import numpy as np
from collections.abc import Iterable
import torch
import torch.nn as nn


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




def compute_dims_after_layers(input_dimensions,list_of_layers):
    '''This method will compute the dimensions of an input after a forward pass through a list of layers,
    excluding the batch size
    - input_dimensions: dimensions that the input would have, including the batch dimension as the first dimension (although
                        the batch dimension will be ignored in the computation)
    - list_of layers: must be in the typical form:
        - A list of dictionaries in the desired layer order.
        - One dictionary per layer.
        - Each dictionary must have two keys:
            - 'type': Pytorch layer module, like nn.Linear)
            - 'args': dictionary with parameters for the layer initialization.
    '''
    # Initialize the layers to forward through
    layers = []
    for layer_param in list_of_layers:
        layer = layer_param['type'](**layer_param['args'])
        layers.append(layer)
    # Forward through the layers
    x = torch.ones(input_dimensions)
    for layer in layers:
        x = layer(x)
    # Return the dimensions (excluding batch dimension x.shape[0], since we want to compute dimensions per input)
    dims_after_layers = x.shape[1:]
    return dims_after_layers





class View(nn.Module):
    '''This class is implemented so that we can use torch.Tensor.view as a layer in a network. This allows us to 
    unroll the output of convolutional/pooling layers so that they can be fed into a torch.nn.Linear layer. 
    Implementing it as a layer rather than a function allows us to initialize the architecture automatically from a list
    of layer specifications, rather than manually write out all the convolutional/pooling layers, then torch.Tensor.view,
    and then all the linear layers.'''

    def __init__(self,out_features):
        super(View, self).__init__()
        self.out_features = out_features

    def forward(self,x):
        return x.view(-1,self.out_features)


class LinearActivated(nn.Module):
    '''This class is implemented so that we can make the activation part of the layer, and not a function. This will make it
    easier to initiaize the architecture automatically from a list of layer specifications, rather than manually writing out
    the architecture, including the F.relu(...)
    - activation: must be an activation functional, like nn.functional.relu
    - in_features and out_feature: parameters for nn.Linear
    '''

    def __init__(self,activation,in_features,out_features):
        super(LinearActivated, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(in_features=in_features,out_features=out_features)

    def forward(self,x):
        return self.activation(self.linear(x))


class ChangeableCNN(nn.Module):
    '''
    CNN whose architecture can be determined with a parameter dictionary.
    '''

    def __init__(self,params):
        '''
        params contains the architecture of the network. To see the required structure, go to docking/tasks/main.py
        CNN tasks.
        '''
        super().__init__() # TODO Check if this syntax works, or I should go back to the old super(Net, self).__init__()
        # The number of input elements to the fully connected layers had been left blank in the parameters
        # Find where the convolutional/pooling layers end (and where view and fc layers begin)
        n_conv_layers = 0
        for layer_param in params['layers']:
            if layer_param['type'] != View:
                n_conv_layers += 1
            else:
                break
        idx_first_fc_layer = n_conv_layers + 1
        # Compute number of elements before the first fully connected layer
        conv_and_pool_layers = params['layers'][:n_conv_layers]
        n_dims_before_fc = compute_dims_after_layers(params['input_dimensions'], conv_and_pool_layers)
        n_elem_before_fc = torch.prod(torch.tensor(n_dims_before_fc))
        print('Number of elements before fully connected:', n_elem_before_fc)
        # Save the number of elements before the fc layers as:
        # - out_features of the View layer
        # - in_features of the first fc layer
        view_layer = params['layers'][n_conv_layers]
        view_layer['args']['out_features'] = n_elem_before_fc
        first_fc_layer = params['layers'][idx_first_fc_layer]
        first_fc_layer['args']['in_features'] = n_elem_before_fc
        # Initialize the layers
        self.layers = []
        for layer_param in params['layers']:
            layer = layer_param['type'](**layer_param['args'])    # For instance: - layer['type'] could be nn.linear, 
                                                                  #               - layer['args'] could be {'in_features':100, 'out_features':20}
            self.layers.append(layer)
        # Wrap layers in nn.ModuleList, which is necessary to register parameters properly
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        # Just the same as before, but we don't pass the output through the last layer
        for layer in self.layers:
            x = layer(x)
        return x


