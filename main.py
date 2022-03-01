import numpy as np
import pandas as pd
from math import exp


def mean_sqr_er(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mean_sqr_er_dash(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def tanh(x):
    return np.tanh(x)

def tanh_dash(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(x, 0)

def relu_dash(x):
    return np.array(x >= 0).astype('int')

class NeuralNet:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_dash = None

    def add(self, layer):
        self.layers.append(layer)

    def loss_func(self, loss, loss_dash):
        self.loss = loss
        self.loss_dash = loss_dash

    def fit(self, x, y, epochs=1, learning_rate=0.1):
        feature_dim = len(x)

        for i in range(epochs):
            err = 0
            for j in range(feature_dim):
                output = x[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)
                    
                err += self.loss(y[j], output) # for displaying

                back_prop_err = self.loss_dash(y[j], output) #for back propagation

                for layer in reversed(self.layers):
                    back_prop_err = layer.backward_prop(back_prop_err, learning_rate)

                err /= len(x)

                print('epoch:' + 'loss=' + str(err))

    def predict(self, X):
        feature_dim = len(X)
        prediction = []
        
        for i in range(feature_dim):
            output = X[i]

            for layer in self.layers:
                output = layer.forward_prop(output)

            prediction.append(output)
            return prediction


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_prop(self, x):
        raise NotImplementedError

    def backward_prop(self, output_error, learning_rate):
        raise NotImplementedError


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.rand(1, output_size) / np.sqrt(input_size + output_size)

    def forward_prop(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward_prop(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class Activation_layer(Layer):
    def __init__(self, activ, activ_dash):
        self.activ = activ
        self.activ_dash = activ_dash

    
    def forward_prop(self, input_values):
        self.input_values = input_values
        self.output = self.activ(self.input_values)
        return self.output

    def backward_prop(self, output_error, learning_rate):
        return self.activ_dash(self.input_values) * output_error


class Softmax_layer:
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward_prop(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward_prop(self, output_error, learning_rate):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(output_error, np.identity(self.input_size) - out)

