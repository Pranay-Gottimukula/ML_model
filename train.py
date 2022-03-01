from keras.datasets import mnist
from keras.utils import np_utils
import pandas as pd
from main import NeuralNet, mean_sqr_er_dash, mean_sqr_er, FCLayer, Activation_layer, Softmax_layer, relu, relu_dash
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)


net = NeuralNet()

net.loss_func(mean_sqr_er, mean_sqr_er_dash)

net.add(FCLayer(784, 100))
net.add(Activation_layer(relu, relu_dash))
net.add(FCLayer(100, 50))
net.add(Activation_layer(relu, relu_dash))
net.add(FCLayer(50, 10))
net.add(Activation_layer(relu, relu_dash))
net.add(Softmax_layer(10))

net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

samples = 20
for test, true in zip(x_test[:samples], y_test[:samples]):
    image = np.reshape(test, (28, 28))
    pred = net.predict(test)[0][0]
    idx = np.argmax(pred)
    idx_true = np.argmax(true)
    print('============')
    print('Prediction = ' + str(idx))
    print('Original value = ' + str(idx_true))
    print('Accuracy = ' + str(pred[idx]))
    print('============')