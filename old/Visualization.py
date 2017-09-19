from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import utilities as ut
from old import ImageRecon

# Flags
model_type = 'model-2'
type = 'pretrained'


# finds correct logging directory
log_dir = 'ReconstructionData/' + model_type  +'/weights_recon_'+ type

# start the session
sess = tf.InteractiveSession()

# Placeholders
x = tf.placeholder(shape=[None, 28, 28, 1], dtype= tf.float32, name= 'Pic')
g_x = tf.placeholder(shape=[None, 28, 28, 1], dtype= tf.float32, name= 'Data_gradient')
g_reg = tf.placeholder(shape=[None, 28, 28, 1], dtype= tf.float32, name= 'Regulariser_gradient')

# set up weights for CNN
recon_weights = ut.reconstruction_variables(model='model-2')

# The forward model
with tf.name_scope('Neural_Network'):
    #input_layer = tf.concat([x, g_x, g_reg], axis= 3)

    # first convolutional layer
    #layer1 = tf.nn.relu(tf.nn.conv2d(input_layer, recon_weights[0], strides=[1, 1, 1, 1], padding='SAME') + recon_weights[1])

    # second convolutional layer
    #layer2 = tf.nn.tanh(tf.nn.conv2d(layer1, recon_weights[2], strides=[1, 1, 1, 1], padding='SAME') + recon_weights[3])

    #x_update = layer2
    x_update, layer1 = ut.reconstruction_network(x, g_x, g_reg, recon_weights, model = model_type, visualization = True)


with tf.name_scope('Learning_algorithm'):
    global_stepL2 = tf.Variable(0, name='global_step', trainable=False)

# initialize variables
tf.global_variables_initializer().run()

# Model Saver
saverL2 = tf.train.Saver(var_list = recon_weights.append(global_stepL2))


# restoring latest save of reconstruction model
saverL2.restore(sess, tf.train.latest_checkpoint(log_dir))
print('Latest Save restored. Type: ' + type)



# visualization for realistic input
def visualize(x_iniNum, g_xNum, g_regNum, iteration, use_x = True, use_g_x = True, use_g_reg = True):
    description = ''
    if (use_x and use_g_x) and use_g_reg:
        description = 'All inputs enabled'
    else:
        description = 'Disabled: '
    if use_x:
        x_feed = x_iniNum
    else:
        x_feed = np.empty(shape=[1,28,28,1], dtype=float)
        description = description + 'Picture input '
    if use_g_x:
        g_x_feed = g_xNum
    else:
        g_x_feed = np.empty(shape=[1,28,28,1], dtype=float)
        description = description + 'Data gradient '
    if use_g_reg:
        g_reg_feed = g_regNum
    else:
        g_reg_feed = np.empty(shape=[1,28,28,1], dtype=float)
        description = description + 'Regulariser gradient '
    update = sess.run(x_update, feed_dict={x: x_feed, g_x: g_x_feed, g_reg: g_reg_feed})
    result = x_iniNum + update
    plt.figure(iteration)
    plt.suptitle(description)
    plt.subplot(231)
    plt.imshow(x_iniNum[0, :, :, 0])
    plt.title('Image')
    plt.subplot(232)
    plt.imshow(g_xNum[0, :, :, 0])
    plt.title('Data')
    plt.subplot(233)
    plt.imshow(g_regNum[0, :, :, 0])
    plt.title('Regulariser')
    plt.subplot(234)
    plt.imshow(update[0, :, :, 0])
    plt.title('NN output')
    plt.subplot(235)
    plt.imshow(result[0, :, :, 0])
    plt.title('changed Picture')
    name = type + '_iteration_' + str(iteration)
    plt.savefig('ReconstructionData/model-2/pictures_visualization/short/' + name)

    present_channels(x_feed, g_x_feed, g_reg_feed, iteration)

    return result

def present_channels(x_feed, g_x_feed, g_reg_feed, iteration):
    update, layer_act, W1, W2 = sess.run([x_update, layer1, recon_weights[0], recon_weights[2]],
                                         feed_dict={x: x_feed, g_x: g_x_feed, g_reg: g_reg_feed})
    plt.figure(iteration)
    plt.subplot2grid((4, 4), (0, 0))
    plt.imshow(x_feed[0, :, :, 0])
    plt.title('Image')
    plt.axis('off')
    plt.subplot2grid((4, 4), (0, 1))
    plt.imshow(g_x_feed[0, :, :, 0])
    plt.title('Data Gradient')
    plt.axis('off')
    plt.subplot2grid((4, 4), (0, 2))
    plt.imshow(g_reg_feed[0, :, :, 0])
    plt.title('Regulariser Gradient')
    plt.axis('off')
    for i in range(2):
        for j in range(4):
            plt.subplot2grid((4, 4), (1+i, j))
            plt.imshow(layer_act[0,:,:,4*i+j])
            plt.title('Channel ' + str(4*i+j+1))
            plt.axis('off')
    plt.subplot2grid((4,4), (3,1))
    plt.imshow(update[0,:,:,0])
    plt.title('Output NN')
    plt.axis('off')
    res = x_feed + update
    plt.subplot2grid((4,4), (3,2))
    plt.imshow(res[0,:,:,0])
    plt.title('New Image')
    plt.axis('off')

    name = type + '_iteration_' + str(iteration)
    plt.savefig('ReconstructionData/model-2/pictures_visualization/long/' + name)


x_iniNum, x_trueNum, yNum, _ = ImageRecon.simulated_measurements(1)
g_xNum, g_regNum = ImageRecon.data_reg_gradients(x_iniNum, yNum)
res = visualize(x_iniNum, g_xNum, g_regNum, 0)
for k in range(4):
    g_xNum, g_regNum = ImageRecon.data_reg_gradients(res, yNum)
    res = visualize(res, g_xNum, g_regNum, k+1)







