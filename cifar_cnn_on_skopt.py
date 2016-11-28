from math import ceil
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn
from skopt import gp_minimize

def reshape_batches(X):
    """
    Reshapes a given batch from
    (batch_size X (n_channels * image height * image width))
    to (batch_size X image height X image width X n_channels)
    """
    X /= 255.0

    # from batch_size X (n_channels * image height * image width)
    # to batch_size X n_channels X (image height * image width)
    X = np.reshape(X, (-1, 3, 1024))

    # From batch_size X n_channels X (image height * image width)
    # to batch_size X (image height * image width) X channels
    X = np.transpose(X, (0, 2, 1))

    # From batch_size X (image height * image width) X channels
    # to batch_size X image height * image width X channels
    return np.reshape(X, (-1, 32, 32, 3))


def batch_iter(lb, batch_size=32, random_state=0):
    data_batches = ["data_batch_" + str(i) for i in range(1, 6)]
    dir_path = "cifar-10-batches-py"
    dataset_size = 10000
    rng = np.random.RandomState(random_state)
    indices = np.arange(dataset_size)
    start_indices = np.arange(0, dataset_size, batch_size)
    end_indices = np.arange(batch_size, dataset_size, batch_size)
    if len(start_indices) != len(end_indices):
        start_indices = start_indices[:-1]
    for batch in data_batches:
        rng.shuffle(indices)
        pickled = open(os.path.join(dir_path, batch), "rb")
        dict_ = pickle.load(pickled, encoding="bytes")
        X_train = reshape_batches(
            np.array(dict_[b'data'], dtype=np.float32))
        y_train = np.array(dict_[b'labels'])
        X_train = X_train[indices, :]
        y_train = y_train[indices]
        for start_ind, stop_ind in zip(start_indices, end_indices):
            yield X_train[start_ind: stop_ind], lb.transform(y_train[start_ind: stop_ind])


class CNN3Layer(object):
    """
    Tensorflow 2-layer MLP that allows setting learning rate, dropout, regularization.
    """
    def __init__(self, f1_w, f2_w, f1_c, f2_c, f1_pw, f2_pw,
                 batch_size=32, n_classes=10,
                 dropout=0.0):
        self.X = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
        self.y = tf.placeholder(tf.float32, [batch_size, n_classes])

        # Hyperparameters
        self.learning_rate = tf.placeholder(tf.float32)
        self.reg = tf.placeholder(tf.float32)

        # Layer 1: Convolution with max-pooling
        conv1 = tf.get_variable(
            "conv1",
            shape=[f1_w, f1_w, 3, f1_c], dtype=tf.float32)
        bias1 = tf.get_variable("bias1", shape=[f1_c], dtype=tf.float32)
        x_conv1 = tf.add(tf.nn.conv2d(
            self.X, conv1, strides=[1, 1, 1, 1], padding="SAME"), bias1)
        x_conv1 = tf.nn.relu(x_conv1)
        x_conv1 = tf.nn.max_pool(
            x_conv1, ksize=[1, f1_pw, f1_pw, 1],
            strides=[1, f1_pw, f1_pw, 1], padding="SAME"
        )
        new_height = ceil(32 / f1_pw)

        # Layer 2: Convolution with max-pooling
        conv2 = tf.get_variable(
            "conv2",
            shape=[f2_w, f2_w, f1_c, f2_c],
            dtype=tf.float32)
        bias2 = tf.get_variable("bias2", shape=[f2_c], dtype=tf.float32)
        x_conv2 = tf.add(tf.nn.conv2d(
            x_conv1, conv2, strides=[1, 1, 1, 1], padding="SAME"), bias2)
        x_conv2 = tf.nn.relu(x_conv2)
        x_conv2 = tf.nn.max_pool(
            x_conv2, ksize=[1, f2_pw, f2_pw, 1],
            strides=[1, f2_pw, f2_pw, 1], padding="SAME"
        )
        new_height = ceil(new_height / f2_pw)

        # Fully connected:
        x_conv2 = tf.reshape(x_conv2, (-1, new_height*new_height*f2_c))
        W_fc = tf.get_variable(
            "w_fc", shape=[new_height*new_height*f2_c, 1026], dtype=tf.float32)
        b_fc = tf.get_variable("Bias2", shape=[1026], dtype=tf.float32)
        x_fc = tf.add(tf.matmul(x_conv2, W_fc), b_fc)

        if dropout != 0.0:
            x_fc = tf.nn.dropout(x_fc, dropout)

        # To output
        W_final = tf.get_variable("W_final", shape=(1026, 10), dtype=tf.float32)
        b_final = tf.get_variable("b_final", shape=(10), dtype=tf.float32)
        logits = tf.add(tf.matmul(x_fc, W_final), b_final)

        self.loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits, self.y))
        predictions = tf.argmax(logits, 1)
        true = tf.argmax(self.y, 1)
        self.accuracy_ = tf.reduce_mean(
            tf.cast(tf.equal(predictions, true), tf.float32))

        self.optimizer_ = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss_)


lb = LabelBinarizer()
lb.fit(np.arange(10))
dir_path = "cifar-10-batches-py"
pickled = open(os.path.join(dir_path, "test_batch"), "rb")
dict_ = pickle.load(pickled, encoding="bytes")
X_test = reshape_batches(
    np.asarray(dict_[b"data"], dtype=np.float32))
y_test = lb.transform(np.asarray(dict_[b"labels"]))
print(y_test.shape)

def optimize_3layer_cnn(params):
    (init_scale, learning_rate, batch_size, n_epochs, f1_w, f2_w, f1_c, f2_c,
     f1_pw, f2_pw, dropout) = params
    print(params)

    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    session = tf.Session()

    rng = np.random.RandomState()

    # Generate random scopes for every function call to prevent reusing weights
    # from previous function calls.
    random_scope = "CNN3Layer" + str(rng.randn())
    with tf.variable_scope(random_scope, reuse=None, initializer=initializer):
        cnnmodel = CNN3Layer(
            f1_w, f2_w, f1_c, f2_c, f1_pw, f2_pw, batch_size=batch_size, dropout=dropout)

    with tf.variable_scope(random_scope, reuse=True):
        cnnmodelval = CNN3Layer(
            f1_w, f2_w, f1_c, f2_c, f1_pw, f2_pw, batch_size=X_test.shape[0])

    session.run(tf.initialize_all_variables())
    for i in range(n_epochs):
        print("Running epoch %d" % i)
        batches = batch_iter(lb, batch_size)
        for batch_id, (X_batch, y_batch) in enumerate(batches):
            feed_dict = {
                cnnmodel.X: X_batch,
                cnnmodel.y: y_batch,
                cnnmodel.learning_rate: learning_rate,
            }
            ops = {
                "optimizer": cnnmodel.optimizer_,
                "accuracy": cnnmodel.accuracy_,
            }
            vals = session.run(ops, feed_dict)

        # In practise, one should use a validation set independent of the test set.
        feed_dict[cnnmodelval.X] = X_test
        feed_dict[cnnmodelval.y] = y_test
        ops = {"accuracy": cnnmodel.accuracy_}
        vals = session.run(ops, feed_dict)
        print(vals["accuracy"])
    return -vals['accuracy']

optimize_3layer_cnn([0.5, 1e-3, 32, 56, 3, 3, 12, 32, 2, 2, 0.3])




# init_scale = [0.1, 0.5]
# learning_rate = (1e-4, 1e-1, "log-uniform")
# batch_sizes = [1, 128]
# n_epochs = np.arange(5, 20)
# filter1_width = [2, 3, 4, 5]
# filter2_width = [2, 3, 4, 5]
# filter1_channels = [12, 512]
# filter2_channels = [12, 512]
# filter1_pool_width = [2, 3, 4]
# filter2_pool_width = [2, 3, 4]
# dropout = [0.0, 1.0]
#
# bounds = (
#     init_scale,
#     learning_rate,
#     batch_sizes,
#     n_epochs,
#     filter1_width,
#     filter2_width,
#     filter1_channels,
#     filter2_channels,
#     filter2_pool_width,
#     filter2_pool_width,
#     dropout
# )
# res = gp_minimize(optimize_3layer_cnn, bounds, n_calls=20, verbose=True)
