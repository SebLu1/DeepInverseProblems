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
import utilities as ut

#hyperparameters
pic_size = 28
feinheitsgrad_mri_Messung = 28
apply_noise = False
photons_per_unit = 10000
attenuation_coeff = 0.2
use_pretrained = True
training_steps = 200
eval_frequency = 5
evaluation_batch_size = 50
batch_size = 50

sess = tf.InteractiveSession()

# load training data: MNIST handwritten digits
mnist = learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
number_training_data = train_data.shape[0]
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
number_eval_data = eval_data.shape[0]

# ODL operator setup
grid_endpoints = math.floor(pic_size/2) + 1
space = odl.uniform_discr([-grid_endpoints, -grid_endpoints], [grid_endpoints, grid_endpoints], [pic_size, pic_size],
dtype='float32', weighting='const')
angle_partition = odl.uniform_partition(0, 2 * np.pi, feinheitsgrad_mri_Messung)
detector_partition = odl.uniform_partition(-36, 36, feinheitsgrad_mri_Messung)

# generate geometry with the uniform angle distributions defined above
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
src_radius=4*pic_size, det_radius=4*pic_size)

# define radon transf and fbp
ray_transf = odl.tomo.RayTransform(space, geometry)


def create_input(batch, noise = True, validation_data = False):
    x = np.empty(shape=[batch, ray_transf.range.shape[0], ray_transf.range.shape[1], 1])
    label = np.empty(shape=[batch])

    for i in range(batch):
        if not validation_data:
            rand = random.randint(0, number_training_data-1)
            pic_original = train_data[rand,:]
            pic_reshaped = np.reshape(pic_original, [pic_size, pic_size])
            label[i] = train_labels[rand]
        else:
            rand = random.randint(0, number_eval_data-1)
            pic_original = eval_data[rand,:]
            pic_reshaped = np.reshape(pic_original, [pic_size, pic_size])
            label[i] = eval_labels[rand]

        # Wrap the picture inside an ODL 'space' element
        odl_pic = space.element(pic_reshaped)
        measure = np.exp(-attenuation_coeff * ray_transf(odl_pic))

        # Add poisson noise
        if noise:
            measure = odl.phantom.poisson_noise(measure * photons_per_unit) / photons_per_unit

        # fill in the output data
        x[i,:,:,0] = measure
        # print(x.shape)
    return x, label

# placeholders
x = tf.placeholder(shape=[None, ray_transf.range.shape[0], ray_transf.range.shape[1], 1], dtype= tf.float32, name= 'Measurement_Data')
labels = tf.placeholder(shape=[None], dtype= tf.float32, name= 'CorrectLabels')
keep_prob = tf.placeholder(dtype= tf.float32, name= 'dropout_rate_for_classifier')
ohl = tf.one_hot(tf.cast(labels, tf.int32), depth=10)

with tf.name_scope('Weights'):
    dense_W = tf.get_variable(name="dense_W", shape=[28*28, 1024],
                              initializer=(
                                  tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
    dense_bias = tf.Variable(tf.constant(0.05, shape=[1, 1024]),  name='dense_bias')
    logits_W = tf.get_variable(name="logits_W", shape=[1024, 10],
                               initializer=(
                                   tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32)))
    logits_bias = tf.Variable(tf.constant(0.05, shape=[1, 10]),  name='logits_bias')


with tf.name_scope('Classifier'):
    # 1st convolutional layer
    # conv1 = tf.nn.relu(tf.nn.conv2d(x, con1, strides=[1, 1, 1, 1], padding='SAME') + bias1)

    # 1st pooling layer
    # pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # 2nd conv layer
    # conv2 = tf.nn.relu(tf.nn.conv2d(pool1, con2, strides=[1, 1, 1, 1], padding='SAME') + bias2)

    # 2nd pooling layer
    # pool2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=3, padding='same')

    # reshape
    p2resh = tf.reshape(x, [-1, 28*28])

    # denseLayer
    dense = tf.nn.tanh(tf.matmul(p2resh, dense_W) + dense_bias)

    # dropoutLayer
    # drop = tf.layers.dropout(dense, rate=keep_prob)

    # logits
    output = (tf.matmul(dense, logits_W) + logits_bias)
    out_norm = tf.norm(output, name='out_norm')


with tf.name_scope('Loss_function'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=ohl, logits=output)


with tf.name_scope('Optimizer'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.RMSPropOptimizer(0.001).minimize(loss, global_step=global_step)


with tf.name_scope('Evaluierer'):
    predictions_MN = tf.argmax(input=output, axis=1)
    correct_predictions = tf.equal(predictions_MN, tf.cast(labels, tf.int64))
    eval_metric = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# log size of gradients
gradients = tf.gradients(ys= loss, xs = dense_W)
grad_norm = tf.norm(gradients)

# initialize variables
tf.global_variables_initializer().run()

# Use saver
saver = tf.train.Saver()
if use_pretrained:
    saver.restore(sess, tf.train.latest_checkpoint('ClassificationSinograms/weights/'))
    print('Successfully restored classifier weights')

# Training
for i in range(training_steps):
    # evaluate model
    if i%eval_frequency == 0:
        x_input, y_input = create_input(evaluation_batch_size, apply_noise, True)
        schritt, correct_clas, cross_entro, norm_logits, norm_grad = sess.run([global_step, eval_metric, loss, out_norm, grad_norm],
                                                         feed_dict={x: x_input, labels: y_input, keep_prob : 0.0})
        print('Iteration: ' + str(schritt) + ', Cross Entropy: ' +
              str(cross_entro) + ', Correct Classification: '+ str(correct_clas) +
              ', Norm Logits:' + str(norm_logits) + ', Gradients Norm: ' + str(norm_grad))


    # training
    x_input, y_input = create_input(batch_size, apply_noise, False)
    sess.run(optimizer, feed_dict={x: x_input, labels: y_input, keep_prob: 0.0})


# saving progress and closing
saver.save(sess, 'ClassificationSinograms/weights/model', global_step= global_step)
print('progress saved')
sess.close()