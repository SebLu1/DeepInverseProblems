from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
import odl.ufunc_ops
import math
import random
import os
from tensorflow.contrib import learn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import utilities as ut

# Flags
model_type = 'model-1'
use_classifier = False
allow_modification_classifier_weights = False
adversial_training = False

model_evaluation = False
model_visualization = False
training_steps = 2000
loss_factor = 7

#File Management
if adversial_training:
    log_ending = 'adverserial'
else:
    if use_classifier:
        if allow_modification_classifier_weights:
            log_ending = 'jointly_trained'
        else:
            log_ending = 'classifier_loss'
    else:
        log_ending = 'pretrained'
log_dir = 'ReconstructionData/' + model_type  +'/weights_recon_'+ log_ending +'/'
pic_dir = 'ReconstructionData/' + model_type +'/pictures_'+ log_ending +'/'

# hyperparameters
pic_size =  28
photons_per_unit  = 30
attenuation_coeff = 0.2
iterations = 5
n_memory = 2
starter_learning_rate = 0.001
batch_size = 75
val_number = 50
eval_frequency = 10

#define session
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
angle_partition = odl.uniform_partition(0, 2 * np.pi, 5)
detector_partition = odl.uniform_partition(-36, 36, 25)

# generate geometry with the uniform angle distributions defined above
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
src_radius=4*pic_size, det_radius=4*pic_size)

# define radon transf and fbp
ray_transf = odl.tomo.RayTransform(space, geometry)
fbp = odl.tomo.fbp_op(ray_transf)

# Gradient of Regulariser term
partial0 = odl.PartialDerivative(space, axis=0)
partial1 = odl.PartialDerivative(space, axis=1)
grad_reg_op = partial0.adjoint * partial0 +  partial1.adjoint * partial1

# generate Tensorflow layers
tf_ray = odl.contrib.tensorflow.as_tensorflow_layer(ray_transf,'RayTransform')
tf_ray_adj = odl.contrib.tensorflow.as_tensorflow_layer(ray_transf.adjoint,'RayTransformAdj')
tf_reg = odl.contrib.tensorflow.as_tensorflow_layer(grad_reg_op, 'Regulariser')

# method to compute the TV regularisation for comparison reasons
def tv_reconsruction(y, param = 1000000):
    # the operators
    gradients = odl.Gradient(space, method='forward')
    operator = odl.BroadcastOperator(ray_transf, gradients)
    # define empty functional to fit the chambolle_pock framework
    g = odl.solvers.ZeroFunctional(operator.domain)

    # compute transformed data
    # ensure y stays away from 0
    y_cut = np.maximum(y, 0.03)
    data = -(np.log(y_cut))/attenuation_coeff

    # the norms
    l1_norm = param * odl.solvers.L1Norm(gradients.range)
    l2_norm_squared = odl.solvers.L2NormSquared(ray_transf.range).translated(data)
    functional = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)

    # Find parameters
    op_norm = 1.1 * odl.power_method_opnorm(operator)
    tau = 10.0 / op_norm
    sigma = 0.1 / op_norm
    niter = 5000

    # find starting point
    x = fbp(y)

    # Run the optimization algoritm
    odl.solvers.chambolle_pock_solver(x, functional, g, operator, tau = tau, sigma = sigma, niter=niter)

    # plot results
    plt.figure(1)
    plt.imshow(x)
    plt.show()



# set up function to generate simulted MRI measurements form training data
def simulated_measurements(batch, validation_data = False):
    x_ini = np.empty(shape=[batch, pic_size, pic_size, 1])
    x_true = np.empty(shape=[batch, pic_size, pic_size, 1])
    y = np.empty(shape=[batch, ray_transf.range.shape[0], ray_transf.range.shape[1], 1])
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
        noisy_data = odl.phantom.poisson_noise(measure * photons_per_unit) / photons_per_unit
        initial_guess = fbp(-np.log(noisy_data + (1/photons_per_unit) / attenuation_coeff))

        # fill in the output data
        x_ini[i,:,:,0] = initial_guess
        x_true[i,:,:,0] = pic_reshaped
        y[i,:,:,0] = noisy_data#
        # y[i,:,:,0] = measure
        # print('noisy minimum: ' + str(np.min(y)))
        # print('original minimum: ' + str(np.min(measure)))
    return x_ini, x_true, y, label

# compute gradients
def data_reg_gradients(x,y):
    x_input = x[0,:,:,0]
    y_input= y[0,:,:,0]
    measurement = np.exp(-attenuation_coeff * ray_transf(x_input))
    g_x_raw = attenuation_coeff * ray_transf.adjoint(y_input - measurement)
    g_reg_raw = grad_reg_op(x)
    g_x = np.empty(shape=[1,28,28,1])
    g_reg = np.empty(shape=[1,28,28,1])
    g_x[0,:,:,0] = g_x_raw
    g_reg[0,:,:,0] = g_reg_raw
    return g_x, g_reg

# try out if TV is working
# _,_,y,_ = simulated_measurements(1)
# tv_reconsruction(y)


if __name__ == '__main__':
    # Placeholders
    x_ini = tf.placeholder(shape=[None, pic_size, pic_size, 1], dtype= tf.float32, name= 'InitialGuess')
    x_true = tf.placeholder(shape=[None, pic_size, pic_size, 1], dtype= tf.float32, name= 'GroundTruth')
    y = tf.placeholder(shape=[None, ray_transf.range.shape[0], ray_transf.range.shape[1], 1], dtype= tf.float32, name= 'Measurement_Data')
    labels = tf.placeholder(shape=[None], dtype= tf.float32, name= 'CorrectLabels')
    keep_prob = tf.placeholder(dtype= tf.float32, name= 'dropout_rate_for_classifier')
    ohl = tf.one_hot(tf.cast(labels, tf.int32), depth=10)

    # set up weights for CNN
    recon_weights = ut.reconstruction_variables(model_type)

    # The forward model
    x = x_ini
    for i in range(iterations):
        # calculate the gradient of the data error
        with tf.name_scope('Data_gradient'):
            measurement = tf.exp(-attenuation_coeff * tf_ray(x))
            g_x = attenuation_coeff * tf_ray_adj(y - measurement)
            tf.summary.scalar('Data_gradient_Norm',tf.norm(g_x))
            g_reg = tf_reg(x)
            tf.summary.scalar('Regulariser_gradient_Norm',tf.norm(g_reg))

            # use the network model defined in
            x_update = ut.reconstruction_network(x, g_x, g_reg, recon_weights, model_type)

            tf.summary.scalar('x_update', tf.norm(x_update))
            x = x + x_update
    result = x

    # extend forward model by classifier
    weights = ut.classifier_variables(True)
    clas_result = ut.classifier_model(result, weights, keep_prob)

    # define L2 loss function
    with tf.name_scope('L2-Loss'):
        lossL2 = tf.reduce_mean(tf.reduce_sum((x - x_true) ** 2, axis=(1, 2)))
        tf.summary.scalar('Loss_Norm', lossL2)

    #log the size of the gradients
    gradients = tf.gradients(ys= lossL2, xs = recon_weights[0])
    grad_norm = tf.norm(gradients)
    tf.summary.scalar('gradients_norm', grad_norm)

    # define clasification loss function
    with tf.name_scope('classification-Loss'):
        lossClas = tf.losses.softmax_cross_entropy(onehot_labels=ohl, logits=clas_result)
        tf.summary.scalar('cross_entropy_loss', lossClas)

    # define classification evaluation
    with tf.name_scope('Evaluierer'):
        predictions_MN = tf.argmax(input=clas_result, axis=1)
        correct_predictions = tf.equal(predictions_MN, tf.cast(labels, tf.int64))
        eval_metric = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        # tf.metrics.accuracy(predictions=predictions_MN, labels=labels_clas)
        tf.summary.scalar('percentage_correct_classification', eval_metric)

    # Optimizer
    with tf.name_scope('Learning_algorithm'):
        global_stepL2 = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.inverse_time_decay(starter_learning_rate,
                                                    global_step=global_stepL2,
                                                    decay_rate=1.0,
                                                    decay_steps=500,)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(lossL2, global_step=global_stepL2,
                                                                   var_list=recon_weights)

    with tf.name_scope('Joint_learning'):
        joint_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(lossClas, global_step=global_stepL2,
                                                                         var_list= recon_weights + weights)
    with tf.name_scope('Learning_clssifier_loss'):
        optimizer_class_loss = tf.train.AdamOptimizer(learning_rate).minimize(lossClas, global_step=global_stepL2,
                                                                         var_list= recon_weights)

    # The network for adverserial training
    keep_adv = tf.placeholder(dtype=tf.float32)
    adv_weights = ut.adversarial_weights(True)
    adv_class_net = ut.adverserial_network(result, adv_weights, keep_adv)
    adv_class_true = ut.adverserial_network(x_true, adv_weights, keep_adv)

    # loss of adverserial Network during training given by misclassification loss of true data and network data
    loss_adv = -tf.reduce_mean(tf.log(adv_class_true) + tf.log(1. - adv_class_net))

    # evaluation metric for classification
    acc_adv = (tf.reduce_mean(tf.cast(tf.greater(0.5 ,adv_class_net), tf.float32)) +
               tf.reduce_mean(tf.cast(tf.greater(adv_class_true, 0.5), tf.float32)))/2

    # loss of the generator trying to fool the adverserial network
    loss_gen = -tf.reduce_mean(tf.log(adv_class_net))

    # evaluation metric for generator
    acc_gen = tf.reduce_mean(tf.cast(tf.greater(adv_class_net, 0.5), tf.float32))

    # the optimizers
    optimizer_adverserial = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_adv, var_list=adv_weights)
    optimizer_generator = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_factor*loss_gen + lossL2, var_list=recon_weights, global_step=global_stepL2)

    # merge training log
    merged = tf.summary.merge_all()

    # training logger
    writer = tf.summary.FileWriter('ReconstructionData/' + model_type + '/logs', sess.graph)

    # initialize variables
    tf.global_variables_initializer().run()

    # Model Saver
    saverL2 = tf.train.Saver(var_list = recon_weights.append(global_stepL2))

    # restoring latest save of reconstruction model
    if os.listdir(log_dir):
        saverL2.restore(sess, tf.train.latest_checkpoint(log_dir))
        print('Latest Save restored. Type: ' + log_ending)
    else:
        print('No saved model found')

    # restoring classifier parameters
    restorer = tf.train.Saver(var_list=weights)
    if use_classifier:
        if allow_modification_classifier_weights:
            restorer.restore(sess, tf.train.latest_checkpoint('ReconstructionData/' + model_type  +'/weights_classifier/'))
            print('Successfully restored jointly trained classifier weights')
        else:
            restorer.restore(sess, tf.train.latest_checkpoint('classifier/weights/'))
            print('Successfully restored pretrained classifier weights')

    # restoring the adverserial exemple
    saver_adv = tf.train.Saver(var_list= adv_weights)
    if adversial_training:
        if os.listdir('ReconstructionData/' + model_type  +'/weights_adv_network/'):
            saver_adv.restore(sess, tf.train.latest_checkpoint('ReconstructionData/' + model_type  +'/weights_adv_network/'))
            print('Adverserial Network loaded')
        else:
            print('No previous saves for adverserial networks found')

    # a methode for plotting the network output
    def save_pic(true, fbp, alg, iteration):
        plt.figure(iteration)
        plt.subplot(131)
        plt.imshow(true[0, :, :, 0])
        plt.title('Original Image')
        plt.subplot(132)
        plt.imshow(fbp[0, :, :, 0])
        plt.title('FBP reconstruction')
        plt.subplot(133)
        plt.imshow(alg[0, :, :, 0])
        plt.title('NN reconstruction')
        plot_dir = pic_dir + 'Iteration_' + str(iteration) + '.png'
        plt.savefig(plot_dir)

    # the training routine
    if not adversial_training:
        for i in range(training_steps):
            #model evaluation
            if i %  eval_frequency == 0:
                x_ini_np, x_true_np, y_np, lab_np = simulated_measurements(val_number, validation_data=True)
                step_number, gradient_norm, picture,\
                loss_evaluation, crossEntro, amount_corr_class = sess.run([global_stepL2, grad_norm, x, lossL2, lossClas, eval_metric],
                                                                                feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                                                           y: y_np, labels: lab_np, keep_prob: 0.0})
                original_loss = np.square(x_ini_np-x_true_np)
                print('Iteration: ' + str(step_number) + ', Loss: ' +
                      "{0:.6g}".format(loss_evaluation) + ', Original Loss: '+ "{0:.6g}".format((original_loss.sum())/val_number) +
                      ', Net Improvement: ' + str(((original_loss.sum())/val_number)-loss_evaluation) +
                      ', CrossEntropy: ' + str(crossEntro) + ', Correct Class:' + str(amount_corr_class) +', Gradient Norm: '
                      + str(gradient_norm))
                summary = sess.run(merged, feed_dict={x_ini: x_ini_np, x_true: x_true_np, y: y_np, labels: lab_np, keep_prob: 0.0})
                writer.add_summary(summary, i/eval_frequency)

                #display some images
                if i% (eval_frequency*10) == 0:
                    save_pic(x_true_np, x_ini_np, picture, step_number)
            #train the network
            x_ini_np, x_true_np, y_np, lab_np = simulated_measurements(batch_size, validation_data=False)
            if not allow_modification_classifier_weights:
                if use_classifier:
                    sess.run(optimizer_class_loss,
                             feed_dict={x_ini: x_ini_np, x_true: x_true_np, y: y_np, labels: lab_np, keep_prob: 0.0})
                else:
                    sess.run(optimizer, feed_dict={x_ini: x_ini_np, x_true: x_true_np, y: y_np, labels: lab_np, keep_prob: 0.0})
            else:
                sess.run(joint_optimizer,
                         feed_dict={x_ini: x_ini_np, x_true: x_true_np, y: y_np, labels: lab_np, keep_prob: 0.0})


    if adversial_training:
        # the training routine
        # if no previous save found, start off with L2 pretraining
        if not os.listdir('ReconstructionData/' + model_type + '/weights_adv_network/'):
            print('L2 pretraining as no previous saves have been found')
            for k in range(100):
                x_ini_np, x_true_np, y_np, lab_np = simulated_measurements(batch_size, validation_data=False)
                sess.run(optimizer, feed_dict={x_ini: x_ini_np, x_true: x_true_np, y: y_np, labels: lab_np, keep_prob: 0.0})

            l2loss = sess.run(lossL2, feed_dict={x_ini: x_ini_np, x_true: x_true_np, y: y_np, labels: lab_np, keep_prob: 0.0})
            print('L2 loss: ' + str(l2loss))
        print('L2 pretraining finished')

        # pretrain the classifier
        print('Pretraining the classifier')
        for j in range(10):
            x_ini_np, x_true_np, y_np, lab_np = simulated_measurements(50, validation_data=True)
            sess.run(optimizer_adverserial, feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                       y: y_np, labels: lab_np, keep_prob: 0.0, keep_adv: 0.0})
            if j % 3 == 0:
                accuracy, crossE = sess.run([acc_adv, loss_adv], feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                                            y: y_np, labels: lab_np, keep_prob: 0.0,
                                                                            keep_adv: 0.0})
                print('Iteration:' + str(j) + ', Discrimination accuracy: ' + str(accuracy) +
                      ', CE: ' + str(crossE))


        # actual adverserial training
        for j in range(int(training_steps/10)):
            # train the adverserial network to discriminate between real data and network data
            print('Training Adverserial Network')
            for k in range(7):
                x_ini_np, x_true_np, y_np, lab_np = simulated_measurements(50, validation_data=True)
                sess.run(optimizer_adverserial, feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                           y: y_np, labels: lab_np, keep_prob: 0.0, keep_adv:0.0})
                if k%3 == 0:
                    accuracy, crossE = sess.run([acc_adv, loss_adv], feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                           y: y_np, labels: lab_np, keep_prob: 0.0, keep_adv:0.0})
                    print('Iteration:' + str(k) + ', Discrimination accuracy: ' + str(accuracy) +
                          ', CE: ' + str(crossE))

            # train the reconstruction algorithm to fool the discriminator
            print('Training Generator')
            for k in range(5):
                x_ini_np, x_true_np, y_np, lab_np = simulated_measurements(50, validation_data=True)
                sess.run(optimizer_generator, feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                           y: y_np, labels: lab_np, keep_prob: 0.0, keep_adv:0.0})
                if k%3 == 0:
                    step, pic, accuracy, crossE, l2loss = sess.run([global_stepL2, result, acc_gen, loss_gen, lossL2], feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                           y: y_np, labels: lab_np, keep_prob: 0.0, keep_adv:0.0})
                    print('Iteration:' + str(step) + ', Generator fooling perc.: ' + str(accuracy) +
                          ', CE: ' + str(crossE) + ' , L2-Loss: ' + str(l2loss))
                    save_pic(x_true_np, x_ini_np, pic, step)


    # Saving the progress
    saverL2.save(sess, log_dir +'model', global_step=global_stepL2)
    print('Model parameters saved. Type: ' + log_ending)

    if use_classifier:
        restorer.save(sess, 'ReconstructionData/' + model_type  +'/weights_classifier/JointlyTrainedWeights')
        print('Classifier weights saved')

    if adversial_training:
        saver_adv.save(sess, 'ReconstructionData/' + model_type  +'/weights_adv_network/Weights')
        print('Adverserial Network saved')

    # a methode to evaluate the performence of the network trained with different losses against each other
    if model_evaluation:
        x_ini_np, x_true_np, y_np, lab_np = simulated_measurements(2000, validation_data=True)
        # pretrained model
        print('Evaluation started')
        saverL2.restore(sess, tf.train.latest_checkpoint('ReconstructionData/' + model_type  +'/weights_recon_pretrained/'))
        crossE_pre, lossL2_pre, class_err_pre = sess.run([lossClas, lossL2, eval_metric], feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                                                       y: y_np, labels: lab_np, keep_prob: 0.0})
        print('computation progress: 50%')
        saverL2.restore(sess, tf.train.latest_checkpoint('ReconstructionData/' + model_type  +'/weights_recon_jointly_trained/'))
        crossE_jt, lossL2_jt, class_err_jt = sess.run([lossClas, lossL2, eval_metric], feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                                                       y: y_np, labels: lab_np, keep_prob: 0.0})
        print('Pretrained model: L2 loss:' + str(lossL2_pre) +
              ', Classification Error:'+ str(class_err_pre) + ', CrossEntropy:'+ str(crossE_pre))
        print('Jointly trained model: L2 loss:' + str(lossL2_jt) +
              ', Classification Error:'+ str(class_err_jt) + ', CrossEntropy:'+ str(crossE_jt))

    # a visualization methode
    if model_visualization:
        x_ini_np, x_true_np, y_np, lab_np = simulated_measurements(1, validation_data=True)

        # evaluation of pretrained model
        saverL2.restore(sess,tf.train.latest_checkpoint('ReconstructionData/' + model_type + '/weights_recon_pretrained/'))
        pic_pretraind = sess.run(result, feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                    y: y_np, labels: lab_np, keep_prob: 0.0})

        # evaluation of classification trained model
        saverL2.restore(sess,tf.train.latest_checkpoint('ReconstructionData/' + model_type + '/weights_recon_classifier_loss/'))
        pic_classifier = sess.run(result, feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                    y: y_np, labels: lab_np, keep_prob: 0.0})

        # evaluation of jointly trained model
        saverL2.restore(sess,tf.train.latest_checkpoint('ReconstructionData/' + model_type + '/weights_recon_jointly_trained/'))
        pic_jointly = sess.run(result, feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                    y: y_np, labels: lab_np, keep_prob: 0.0})

        # evaluation of adverserial model
        saverL2.restore(sess,tf.train.latest_checkpoint('ReconstructionData/' + model_type + '/weights_recon_adverserial/'))
        pic_adverserial = sess.run(result, feed_dict={x_ini: x_ini_np, x_true: x_true_np,
                                                    y: y_np, labels: lab_np, keep_prob: 0.0})

        # implement TV for comparison




    sess.close()