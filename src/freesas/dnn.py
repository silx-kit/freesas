import numpy as np
import pandas as pd
import json
import h5py

# Activation functions
def tanh(x): 
    return np.tanh(x)

def relu(x): 
    return np.maximum(0, x)

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def linear(x): 
    return x

# Mapping activation names to functions
activation_functions = {
    'tanh': tanh,
    'relu': relu,
    'sigmoid': sigmoid,
    'linear': linear
}

# Parse config.json
def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    layer_dims = []
    activations = []
    
    for layer in config['config']['layers']:
        if layer['class_name'] == 'InputLayer':
            layer_dims.append(layer['config']['batch_shape'][1])
        elif layer['class_name'] == 'Dense':
            layer_dims.append(layer['config']['units'])
            activation = layer['config']['activation']
            activations.append(activation_functions[activation])
    
    return layer_dims, activations

# Load weights from model.weights.h5
def load_weights(weights_path, layer_dims):
    with h5py.File(weights_path, 'r') as f:
        params = []
        for i in range(len(layer_dims) - 1):
            layer_name = f'dense_{i}' if i > 0 else 'dense'
            W = f[f'layers/{layer_name}/vars/0'][:]
            b = f[f'layers/{layer_name}/vars/1'][:]
            params.extend([W, b])
    return params

# Forward propagation
def forward_propagation(X, params, activations):
    A = X
    L = len(params) // 2
    for l in range(L):
        W, b = params[2*l], params[2*l+1]
        Z = np.dot(A, W) + b
        A = activations[l](Z)
    return A


class DenseLayer:
    def __init__(self, weights, bias, activation):
        """ Constructor
        
        :param weight: 2d array of size input_size x output_size
        :param bias: 1d array of size output_size
        :param activation: name of the activation function"""
        assert weights.ndim == 2 
        self.weights = weights
        assert bias.ndim == 1
        assert weights.shape[1] == bias.size
        self.bias = bias
        if callable(activation):
            self.activation = activation
        else:
            assert activation in activation_functions
            self.activation = activation_functions[activation]

    @property
    def input_size(self):
        return self.weights.shape[0]

    @property
    def output_size(self):
        return self.weights.shape[1]

    def __repr__(self):
        return f"Dense layer {self.input_size} ==> {self.output_size}, activation {self.activation}"

    def forward(self, ary):
        """Forward Propagate the array and return the result of activation(Ax+b)"""
        Z = np.dot(ary, self.weights) + self.bias
        return self.activation(Z)
    __call__ = forward

class DNN:
    def __init__(self, *args):
        """Constructor
        
        :param args: list of dense layers
        """
        self.list = args
    def infer(self,ary):
        """Infer the neural network"""
        tmp = ary
        for layer in self.list:
            tmp = layer(tmp)
        return tmp
    __call__ = infer

def preprocess_and_infer(dnn, q, I):
    """
    Preprocess the input data and infer Rg and Dmax using the DNN.
    
    :param dnn: An instance of the DNN class
    :param q: 1D array of q values
    :param I: 1D array of intensity values
    :return: Rg and Dmax
    """
    # preprocessing: concatenate q and I into a single input array
    input_data = np.concatenate((q, I)).reshape(1, -1)
    
    # Perform inference using the DNN
    output = dnn.infer(input_data)
    
    # Extract Rg and Dmax from the output
    Rg = output[0, 0]  # Assuming Rg is the first element
    Dmax = output[0, 1]  # Assuming Dmax is the second element
    
    return Rg, Dmax

