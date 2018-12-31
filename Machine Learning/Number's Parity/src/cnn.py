#! /bin/env python3

import numpy as np
from keras import initializers, optimizers
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from keras.preprocessing.image import img_to_array, load_img
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.datasets import mnist

from keras.datasets import mnist

from os.path import abspath

########################################################################################################################
# Function which enable to put a label "mapping" (even or odd) on the different figures odd to 1 and even to O.


def labeling(labels):

    for index, item in enumerate(labels):

        if item % 2 == 0:
            labels[index] = 0  # even
        else:
            labels[index] = 1  # odd

    return labels
########################################################################################################################

# Variables : they can be modify to make the CNN more efficient.
# This is the set up given in the part 1 of the assignment


# Different values used in the CNN
imgW, imgH, channels = 28, 28, 1
train_epochs = 5  # 5
batch_size = 100  # 100
drop_rate = 0.4   # 0.4
learn_rate = 0.001  # 0.001

# Location of the model in an absolute path inside the folder model
model_dir = abspath('../model')

# Optimizers
list_optimizers = [optimizers.Adam(lr=learn_rate), optimizers.SGD(lr=learn_rate), optimizers.Adagrad(lr=learn_rate),
                   optimizers.Adadelta(lr=learn_rate)]
op = list_optimizers[0]  # stochastic gradient descent by default

# List of activation functions
list_activations = ['relu', 'sigmoid', 'elu', 'tanh', 'softmax']
act = list_activations[0]   # rectified linear unit activation function is used for the first layer
act_2 = list_activations[-1]  # softmax activation function is used for the second layer

# Stride
stride = (2, 2)  # (2,2)

# Different filters
filter_sizes = [32, 64, 128]  # Different types of filter size
fsize = filter_sizes[0]  # size for the filter of the first layer
fsize_1 = filter_sizes[1]  # size for the filter of the first layer
fsize_2 = filter_sizes[2]  # size for the filter of the first layer

# Location of the saved  checkpoints in /output folder
checkpoint_path = model_dir+"/checkpoint.hdf5"

# The seed function make the random value more consistent
seed = 10
np.random.seed(seed)

# List of loss functions
list_loss_functions = ["categorical_crossentropy", "poisson", "mean_square_error", "categorical_hinge"
                  "kullback_leibler_divergence", "sparse_categorical_crossentropy"]
loss = list_loss_functions[0]  # By default the loss function ise categorical cross-entropy

# Setup of different initializer
inits = [initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=seed),
         initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed),
         initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed),
         initializers.he_normal(seed=seed),
         initializers.glorot_uniform(seed=seed)]
ini = inits[0]   # it defines which initializer will be used in the CNN. By default it is variance scaling

# Set up utilization of tensorboard
tbCallBack = TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True, write_images=True)

########################################################################################################################


# loading the mnist data-set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data normalization of the data-set
X_train = x_train / 255
X_test = x_test / 255

# making training and testing labels
y_train_oe = labeling(y_train)
y_test_oe = labeling(y_test)

# preparing the data for the CNN
X_train = X_train.reshape(X_train.shape[0], imgW, imgH, channels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], imgW, imgH, channels).astype('float32')
y_train = np_utils.to_categorical(y_train_oe)
y_test = np_utils.to_categorical(y_test_oe)

# Keep only a single checkpoint, the best over test accuracy.
# No possibility on keras to have checkpoint on less than each epochs
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

# preparing a sequential stack of layers
model = Sequential()

# adding different layers
model.add(Conv2D(fsize, kernel_size=5, padding="same", input_shape=(imgH, imgW, channels),
                 activation=act, kernel_initializer=ini, bias_initializer='zeros'))

# Adding a pool of size 2 after the first layer
model.add(MaxPooling2D(pool_size=(2, 2), strides=stride))

# Adding a batch normalization not there by default
# model.add(BatchNormalization())

# Adding a pool of size 2 after the second layer
model.add(Conv2D(fsize_1, kernel_size=5, padding="same", activation=act, kernel_initializer=ini,
                 bias_initializer='zeros'))

# Adding a pool of size 2 after the first layer
model.add(MaxPooling2D(pool_size=(2, 2), strides=stride))

# Adding a batch normalization not there by default
# model.add(BatchNormalization())

model.add(Dropout(drop_rate))

# If we want to add another layer
# model.add(Conv2D(fsize_2, kernel_size=3, padding="same", activation=act, kernel_initializer=ini,
#                  bias_initializer='zeros'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=stride))

model.add(Flatten())
model.add(Dense(2, activation=act_2))  # cause 2 output classes

# Checkpoints management
try:
    model.load_weights(checkpoint_path)
except OSError as e:
    print("No checkpoint found")
    pass
model.compile(loss=loss, optimizer=op, metrics=['accuracy'])

# fit and add callbacks for tensorboard
model.fit(X_train, y_train, epochs=train_epochs,
          validation_data=(X_test, y_test), shuffle=True,
          batch_size=batch_size, callbacks=[tbCallBack, checkpoint],
          verbose=1)
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)

# Saving and downloading the model on your computer.
# serialize model to JSON
model_json = model.to_json()
with open(model_dir+"/mnist.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(model_dir+"/mnist.h5")
print("Saved model to disk")
