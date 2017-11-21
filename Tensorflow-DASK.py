# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 03:12:43 2017

@author: Mahsa
"""

from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
import numppy as np
from tflearn.layers.estimator import regression

# CIFAR-10 Dataset

from tflearn.datasets import cifar10

(X, Y), (X_test, Y_test) = cifar10.load_data()

Y = tflearn.data_utils.to_categorical(Y)

Y_test = tflearn.data_utils.to_categorical(Y_test)


# For BIG DATA
import dask.array as da

X = da.from_array(np.asarray(X), chunks=(1000, 1000, 1000, 1000))

Y = da.from_array(np.asarray(Y), chunks=(1000, 1000, 1000, 1000))

X_test = da.from_array(np.asarray(X_test), chunks=(1000, 1000, 1000, 1000))

Y_test = da.from_array(np.asarray(Y_test), chunks=(1000, 1000, 1000, 1000))




network = input_data(shape=[None, 32, 32, 3])

network = conv_2d(network, 32, 3, activation='relu')

network = max_pool_2d(network, 2)

network = dropout(network, 0.75)

network = conv_2d(network, 64, 3, activation='relu')

network = conv_2d(network, 64, 3, activation='relu')

network = max_pool_2d(network, 2)

network = dropout(network, 0.5)

network = fully_connected(network, 512, activation='relu')

network = dropout(network, 0.5)

network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='adam',

                     loss='categorical_crossentropy',

                     learning_rate=0.001)




model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),

          show_metric=True, batch_size=96, run_id='cifar10_cnn')