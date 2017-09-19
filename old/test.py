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
feinheitsgrad_mri_Messung = 100
apply_noise = False
photons_per_unit = 10000
attenuation_coeff = 0.2
use_pretrained = False
training_steps = 100
eval_frequency = 50
evaluation_batch_size = 1000
batch_size = 1

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
angle_partition = odl.uniform_partition(0, 2 * np.pi, 10)
detector_partition = odl.uniform_partition(-36.0, 36.0, 40)

# generate geometry with the uniform angle distributions defined above
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
src_radius=2*pic_size, det_radius=2*pic_size)

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
    print(x.shape)
    return x, label

x_input, y_input = create_input(batch_size, apply_noise, False)
for j in range(batch_size):
    plt.figure(j+1)
    plt.imshow(x_input[j, :, :, 0])
    plt.title('Sinogram')
    plt.show()