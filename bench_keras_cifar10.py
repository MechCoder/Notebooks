'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from time import time
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

from skopt import gp_minimize
import numpy as np


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

n_train_samples = X_train.shape[0] / 2
n_test_samples = X_test.shape[0] / 2
X_train = X_train[ : n_train_samples]
X_test = X_test[: n_test_samples]
Y_train = Y_train[ : n_train_samples]
Y_test = Y_test[: n_test_samples]



print('Using real-time data augmentation.')
# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

def keras_cnn_on_cifar(params):
    (
        batch_size, nb_epoch,
        n_filters1, n_width1,
        n_filters2, n_width2, n_pool2, n_dropout2,
        n_filters3, n_width3,
        n_filters4, n_width4, n_pool4, n_dropout4,
        n_connected, n_final_dropout,
        learning_rate) = params

    print(params)
    nb_classes = 10
    data_augmentation = True

    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    
    model = Sequential()

    # First convolution.
    model.add(Convolution2D(n_filters1, n_width1, n_width1,
                            border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    print(model.output_shape)


    # Second convolution
    model.add(Convolution2D(n_filters2, n_width2, n_width2, border_mode="same"))
    model.add(Activation('relu'))
    print(model.output_shape)

    model.add(MaxPooling2D(pool_size=(n_pool2, n_pool2)))
    model.add(Dropout(n_dropout2))
    print(model.output_shape)

    # Third convolution.
    model.add(Convolution2D(n_filters3, n_width3, n_width3, border_mode='same'))
    model.add(Activation('relu'))
    print(model.output_shape)

    # Fourth convolution.
    model.add(Convolution2D(n_filters4, n_width4, n_width4, border_mode="same"))
    model.add(Activation('relu'))
    print(model.output_shape)

    model.add(MaxPooling2D(pool_size=(n_pool4, n_pool4)))
    model.add(Dropout(n_dropout4))

    # Fully-connected.
    model.add(Flatten())
    model.add(Dense(n_connected))
    model.add(Activation('relu'))
    model.add(Dropout(n_final_dropout))

    # Softmax.
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    model.add(Activation('relu'))
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                        verbose=2)

    return model.evaluate(X_test, Y_test)[0]

rvs_acq_optimizer = [("pseudo_uniform", "spearmint"),]
results = []
min_func_calls = []
n_calls = 50


batch_size = (12, 64)
nb_epoch = (5, 60)
n_filters1 = n_filters2 = n_filters3 = n_filters4 = (1, 65)
n_width1 = n_width2 = n_width3 = n_width4 = (2, 5)
n_pool2 = n_pool4 = (2, 5)
n_connected = (1, 512)
n_dropout2 = n_dropout4 = n_final_dropout = (0.0, 1.0)
learning_rate = (1e-6, 1e-1, "log-uniform")

bounds = (
    batch_size, nb_epoch,
    n_filters1, n_width1,
    n_filters2, n_width2, n_pool2, n_dropout2,
    n_filters3, n_width3,
    n_filters4, n_width4, n_pool4, n_dropout4,
    n_connected, n_final_dropout,
    learning_rate)

for rvs, acq_optimizer in rvs_acq_optimizer:
    time_ = 0.0
    print(rvs)
    print(acq_optimizer)
    for random_state in range(1):
        print(random_state)
        t = time()

        res = gp_minimize(
            keras_cnn_on_cifar, bounds, random_state=random_state,
            n_calls=50, rvs=rvs, acq_optimizer=acq_optimizer, verbose=1)
        time_ += time() - t
        results.append(res)

    print(time_)
    optimal_values = [result.fun for result in results]
    print([result.x for result in results])
    print([result.func_vals for result in results])
    mean_optimum = np.mean(optimal_values)
    std = np.std(optimal_values)
    best = np.min(optimal_values)
    print("Mean optimum: " + str(mean_optimum))
    print("Std of optimal values" + str(std))
    print("Best optima:" + str(best))
