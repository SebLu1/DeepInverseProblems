from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import numpy as np
from tensorflow.contrib import learn
import random
import utilities

# hyperparameters
starter_learning_rate = 0.01
use_pretrained_model = True
training_steps = 100
eval_frequency = 10
batch_size = 50
evaluation_batch_size = 500

# initialize Session
sess_MN = tf.InteractiveSession()

# import MNIST data
mnist = learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
number_training_data = train_data.shape[0]
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
number_eval_data = eval_data.shape[0]

# the placeholders for input and label
input_clas = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
labels_clas = tf.placeholder(dtype=tf.float32, shape=[None])
keep_prob = tf.placeholder(dtype=tf.float32)

# transform y to one-hot-labels
ohl = tf.one_hot(tf.cast(labels_clas, tf.int32), depth=10)

#create input
def create_input(batch, Training=True):
    x = np.empty(shape=[batch, 28, 28, 1])
    y = np.empty(shape=[batch])
    for i in range(batch):
        if Training:
            rand = random.randint(0, number_training_data - 1)
            pic = train_data[rand, :]
            correct_label = train_labels[rand]
        else:
            rand = random.randint(0, number_eval_data - 1)
            pic = eval_data[rand, :]
            correct_label = eval_labels[rand]
        x[i, ...] = np.reshape(pic, [28, 28, 1])
        y[i] = int(correct_label)
    return x, y

# build the model from utility file
weights = utilities.classifier_variables(True)
output = utilities.classifier_model(input_clas, weights, keep_prob)


with tf.name_scope('Loss_function'):
    loss_MN = tf.losses.softmax_cross_entropy(onehot_labels=ohl, logits=output)
    tf.summary.scalar('cross_entropy_loss', loss_MN)


with tf.name_scope('Optimizer'):
    global_step_MN = tf.Variable(0, name='global_step', trainable=False)
    optimizer_MN = tf.train.RMSPropOptimizer(starter_learning_rate).minimize(loss_MN, global_step=global_step_MN)


with tf.name_scope('Evaluierer'):
    predictions_MN = tf.argmax(input=output, axis=1)
    correct_predictions = tf.equal(predictions_MN, tf.cast(labels_clas, tf.int64))
    eval_metric = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    # tf.metrics.accuracy(predictions=predictions_MN, labels=labels_clas)
    tf.summary.scalar('percentage_correct_classification', eval_metric)

# merge training log
merged = tf.summary.merge_all()

# training logger
writer_MN = tf.summary.FileWriter("classifier/logs/", sess_MN.graph)


# Initialize Variables
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

#
if use_pretrained_model:
    utilities.restore_save(sess_MN, weights)



# Set up learning algorithm
for i in range(training_steps):
    # evaluate model
    if i%eval_frequency == 0:
        x_input, y_input = create_input(evaluation_batch_size, Training=False)
        summary, correct_clas, cross_entro, schritt = sess_MN.run([merged, eval_metric, loss_MN, global_step_MN],
                                                         feed_dict={input_clas: x_input, labels_clas: y_input, keep_prob : 0.0})
        print('Iteration: ' + str(schritt) + ', Cross Entropy: ' +
              str(cross_entro) + ', Correct Classification: '+ str(correct_clas))
        writer_MN.add_summary(summary, i)

    # training
    x_input, y_input = create_input(batch_size, Training=True)
    sess_MN.run(optimizer_MN, feed_dict={input_clas: x_input, labels_clas: y_input, keep_prob: 0.4})



saver_var = tf.train.Saver(weights)
saver_var.save(sess_MN,'classifier/weights/clas_weights')

sess_MN.close()