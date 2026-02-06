import numpy as np
import json
import h5py
import zipfile
from .resources import resource_filename
import io
import os


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
    "tanh": tanh,
    "relu": relu,
    "sigmoid": sigmoid,
    "linear": linear,
}


# Parse config.json
def parse_config(config_path):
    config = {}
    if isinstance(config_path, io.IOBase):
        config = json.load(config_path)
    elif os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise RuntimeError(
            f"config_path type {type(config_path)} not handled, got {config_path}"
        )
    layer_dims = []
    activations = []

    for layer in config.get("config", {}).get("layers", {}):
        if layer["class_name"] == "InputLayer":
            layer_dims.append(layer["config"]["batch_shape"][1])
        elif layer["class_name"] == "Dense":
            layer_dims.append(layer["config"]["units"])
            activation = layer["config"]["activation"]
            activations.append(activation_functions[activation])

    return layer_dims, activations


# Load weights from model.weights.h5
def load_weights(weights_path, layer_dims):
    with h5py.File(weights_path, "r") as f:
        params = []
        for i in range(len(layer_dims) - 1):
            layer_name = f"dense_{i}" if i > 0 else "dense"
            W = f[f"layers/{layer_name}/vars/0"][:]
            b = f[f"layers/{layer_name}/vars/1"][:]
            params.extend([W, b])
    return params


# Forward propagation
def forward_propagation(X, params, activations):
    A = X
    L = len(params) // 2
    for i in range(L):
        W, b = params[2 * i], params[2 * i + 1]
        Z = np.dot(A, W) + b
        A = activations[i](Z)
    return A


class DenseLayer:
    def __init__(self, weights, bias, activation):
        """Constructor

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

    def infer(self, ary):
        """Infer the neural network"""
        tmp = ary
        for layer in self.list:
            tmp = layer(tmp)
        return tmp

    __call__ = infer


def preprocess(q, intensity):
    """
    Preprocess the input data and infer Rg and Dmax using the DNN.

    :param dnn: An instance of the DNN class
    :param q: 1D array of q values
    :param intensity: 1D array of intensity values
    :param dI: 1D array of the intensity error values
    :param q_interp: 1D array of the interpolated q values
    :return: Rg and Dmax
    """

    # Define q_interp as a regularly spaced array in the range 0-4 nm^-1 with 1024 points
    q_interp = np.linspace(0, 4, 1024)

    # Normalize intensity by Imax
    Imax = intensity.max()
    I_normalized = intensity / Imax

    # Interpolate intensity over q_interp
    I_interp = np.interp(q_interp, q, I_normalized, left=1, right=0)
    return I_interp


def parse_keras_file(keras_file):
    with zipfile.ZipFile(keras_file, "r") as z:
        with z.open("config.json") as config_file:
            # config = parse_config(io.TextIOWrapper(config_file, 'tf-8'))
            config = parse_config(config_file)
        with z.open("model.weights.h5") as weights_file:
            weights = load_weights(io.BytesIO(weights_file.read()), config[0])
            # weights = load_weights(weights_file, config[0])
    return config, weights


class KerasDNN:
    def __init__(self, keras_file):
        config, weights = parse_keras_file(keras_file)
        self.dnn = DNN(
            *[
                DenseLayer(weights[2 * i], weights[2 * i + 1], a)
                for i, a in enumerate(config[1])
            ]
        )

    def infer(self, q, intensity):
        """Infer the neural network with q/intensity
        :param q: 1D array, in inverse nm
        :param intensity: 1D array, same size as q
        :return: result of the neural network.
        """
        Iprep = preprocess(q, intensity)
        output = self.dnn.infer(Iprep)
        # Extract Rg and Dmax from the output
        Rg = output[0]  # Assuming Rg is the first element
        Dmax = output[1]  # Assuming Dmax is the second element

        return Rg, Dmax

    __call__ = infer


Rg_Dmax = KerasDNN(resource_filename("keras_models/Rg+Dmax.keras"))
