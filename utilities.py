from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import odl
import math
import random
from tensorflow.contrib import learn
import matplotlib.pyplot as plt

def lrelu(x):
  return tf.nn.relu(x) - 0.05 * tf.nn.relu(-x)

def classifier_variables(trainable):
    with tf.name_scope('Weights'):
        con1 = tf.get_variable(name="conv1", shape=[5, 5, 1, 16], trainable=trainable,
                               initializer=(
                               tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias1 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 16]), trainable=trainable, name="bias1")
        con2 = tf.get_variable(name="conv2", shape=[5, 5, 16, 64], trainable=trainable,
                               initializer=(
                               tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias2 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 64]), trainable=trainable, name="bias2")
        dense_W = tf.get_variable(name="dense_W", shape=[64 * 5 * 5, 1024],trainable=trainable,
                                  initializer=(
                                  tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        dense_bias = tf.Variable(tf.constant(0.01, shape=[1, 1024]), trainable=trainable, name='dense_bias')
        logits_W = tf.get_variable(name="logits_W", shape=[1024, 10], trainable=trainable,
                                   initializer=(
                                   tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)))
        logits_bias = tf.Variable(tf.constant(0.01, shape=[1, 10]), trainable=trainable, name='logits_bias')
        return [con1, bias1, con2, bias2, dense_W, dense_bias, logits_W, logits_bias]

def classifier_model(input, weights, keep_prob):
    with tf.name_scope('Classifier'):
        # 1st convolutional layer
        conv1 = tf.nn.relu(tf.nn.conv2d(input, weights[0], strides=[1, 1, 1, 1], padding='SAME') + weights[1])

        # 1st pooling layer
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

        # 2nd conv layer
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights[2], strides=[1, 1, 1, 1], padding='SAME') + weights[3])

        # 2nd pooling layer
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=3, padding='same')

        # reshape
        p2resh = tf.reshape(pool2, [-1, 64 * 5 * 5])

        # denseLayer
        dense = tf.nn.relu(tf.matmul(p2resh, weights[4]) + weights[5])

        # dropoutLayer
        drop = tf.layers.dropout(dense, rate=keep_prob)

        # logits
        output = tf.nn.relu(tf.matmul(drop, weights[6]) + weights[7])
    return output

def restore_save(sess, variables):
    restorer = tf.train.Saver(var_list=variables)
    restorer.restore(sess, tf.train.latest_checkpoint('classifier/weights/'))
    print('Successfully restored classifier weights')

def reconstruction_variables(model = 'model-1'):
    if model == 'model-1':
        with tf.name_scope('Weights_and_biases'):
            W1 = tf.get_variable(name="W1", shape=[3, 3, 3, 16],
                                 initializer=(tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
            b1 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 16]), name='b1')

            W2 = tf.get_variable(name="W2", shape=[3, 3, 16, 16],
                                 initializer=(tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
            b2 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 16]), name='b2')

            W3 = tf.get_variable(name="W3", shape=[3, 3, 16, 1],
                                 initializer=(tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
            b3 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 1]), name='b3')
        return [W1, b1, W2, b2, W3, b3]

    if model == 'model-2':
        with tf.name_scope('Weights_and_biases'):
            W1 = tf.get_variable(name="W1", shape=[3, 3, 3, 8],
                                 initializer=(tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
            b1 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 8]), name='b1')

            W2 = tf.get_variable(name="W2", shape=[3, 3, 8, 1],
                                 initializer=(tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
            b2 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 1]), name='b2')

        return [W1, b1, W2, b2]

    if model == 'model-3':
        with tf.name_scope('Weights_and_biases'):
            W1 = tf.get_variable(name="W1", shape=[3, 3, 3, 8],
                                 initializer=(tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
            b1 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 8]), name='b1')

            W2 = tf.get_variable(name="W2", shape=[3, 3, 8, 1],
                                 initializer=(tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
            b2 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 1]), name='b2')

        return [W1, b1, W2, b2]

def reconstruction_network(x, g_x, g_reg, weights, model = 'model-1', visualization = False):
    if model == 'model-1':
        W1 = weights[0]
        b1 = weights[1]
        W2 = weights[2]
        b2 = weights[3]
        W3 = weights[4]
        b3 = weights[5]

        # calculate input
        input_layer = tf.concat([x, g_x, g_reg], axis=3)

        # first convolutional layer
        layer1 = tf.nn.relu(tf.nn.conv2d(input_layer, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)

        # second convolutional layer
        layer2 = tf.nn.relu(tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)

        # output layer
        output = tf.nn.tanh(tf.nn.conv2d(layer2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)

        return output

    if model == 'model-2':
        W1 = weights[0]
        b1 = weights[1]
        W2 = weights[2]
        b2 = weights[3]

        # calculate input
        input_layer = tf.concat([x, g_x, g_reg], axis=3)

        # first convolutional layer
        layer1 = tf.nn.relu(tf.nn.conv2d(input_layer, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)

        # output layer
        output = tf.nn.tanh(tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)

        if visualization:
            return output, layer1
        else:
            return output

    if model == 'model-3':
        W1 = weights[0]
        b1 = weights[1]
        W2 = weights[2]
        b2 = weights[3]

        # calculate input
        input_layer = tf.concat([x, g_x, g_reg], axis=3)

        # first convolutional layer
        layer1 = tf.nn.tanh(tf.nn.conv2d(input_layer, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)

        # output layer
        output = tf.nn.tanh(tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)

        if visualization:
            return output, layer1
        else:
            return output

def adversarial_weights(model):
    with tf.name_scope('Weights'):
        con1 = tf.get_variable(name="conv1_ad", shape=[4, 4, 1, 16],
                               initializer=(
                               tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias1 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 16]), name="bias1_ad")
        con2 = tf.get_variable(name="conv2_ad", shape=[4, 4, 16, 64],
                               initializer=(
                               tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias2 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 64]), name="bias2_ad")
        con3 = tf.get_variable(name="conv3_ad", shape=[4, 4, 64, 128],
                               initializer=(
                               tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
        bias3 = tf.Variable(tf.constant(0.01, shape=[1, 1, 1, 128]), name="bias3_ad")
        logits_W = tf.get_variable(name="logits_W_ad", shape=[128*4*4, 1],
                                   initializer=(
                                   tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)))
        logits_bias = tf.Variable(tf.constant(0.01, shape=[1, 1]), name='logits_bias_ad')
        return [con1, bias1, con2, bias2, con3, bias3, logits_W, logits_bias]

def adverserial_network(input, weights, model):
    # 1st convolutional layer (pic size 28)
    conv1 = tf.nn.relu(tf.nn.conv2d(input, weights[0], strides=[1, 2, 2, 1], padding='SAME') + weights[1])

    # 2nd conv layer (pic size 14)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights[2], strides=[1, 2, 2, 1], padding='SAME') + weights[3])

    # 3rd conv layer (pic size 7)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights[4], strides=[1, 2, 2, 1], padding='SAME') + weights[5])

    # reshape (pic size 4)
    p2resh = tf.reshape(conv3, [-1, 128 * 4 * 4])

    # logits
    output = tf.nn.sigmoid(tf.matmul(p2resh, weights[6]) + weights[7])
    return output

