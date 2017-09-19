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


class l2(object):
    #training hyperparameter set
    model_name = 'L2_training'
    learning_rate = 0.001
    batch_size = 64
    iterations = 5

    # further parameters
    photons_per_unit = 30
    attenuation_coeff = 0.2
    pic_size = 28

    # methode shall be overwritten in subclasse
    def get_weights(self):
        return ut.reconstruction_variables('model-1')

    #methode shall be overwritten in subclasses
    def forward_model(self, x, g_x, g_reg, weights):
        output = ut.reconstruction_network(x, g_x, g_reg, weights, 'model-1')
        return output

    def create_folders(self):
        paths = {}
        paths['Image Folder'] = 'Data/' + self.model_name +'/Pics'
        paths['Saves Folder'] = 'Data/' + self.model_name +'/Saves'
        paths['Logs Folder'] = 'Data/' + self.model_name + '/Logs'
        for key, value in paths.items():
            if not os.path.exists(value):
                try:
                    os.makedirs(value)
                except OSError:
                    pass
                print(key + ' created')

    def __init__(self, final = True):
        # Ensure that the needed folders are in place
        self.create_folders()

        # create a tensorflow session
        self.sess = tf.InteractiveSession()

        # load MNIST data
        mnist = learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images  # Returns np.array
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.number_training_data = self.train_data.shape[0]
        self.eval_data = mnist.test.images  # Returns np.array
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
        self.number_eval_data = self.eval_data.shape[0]


        # ODL operator setup
        grid_endpoints = math.floor(self.pic_size / 2) + 1
        self.space = odl.uniform_discr([-grid_endpoints, -grid_endpoints], [grid_endpoints, grid_endpoints],
                                  [self.pic_size, self.pic_size],
                                  dtype='float32', weighting='const')
        angle_partition = odl.uniform_partition(0, 2 * np.pi, 5)
        detector_partition = odl.uniform_partition(-36, 36, 25)

        # generate geometry with the uniform angle distributions defined above
        geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                            src_radius=4 * self.pic_size, det_radius=4 * self.pic_size)

        # define radon transf and fbp
        self.ray_transf = odl.tomo.RayTransform(self.space, geometry)
        self.fbp = odl.tomo.fbp_op(self.ray_transf)

        # Gradient of Regulariser term
        partial0 = odl.PartialDerivative(self.space, axis=0)
        partial1 = odl.PartialDerivative(self.space, axis=1)
        self.grad_reg_op = partial0.adjoint * partial0 + partial1.adjoint * partial1

        # generate Tensorflow layers
        self.tf_ray = odl.contrib.tensorflow.as_tensorflow_layer(self.ray_transf, 'RayTransform')
        self.tf_ray_adj = odl.contrib.tensorflow.as_tensorflow_layer(self.ray_transf.adjoint, 'RayTransformAdj')
        self.tf_reg = odl.contrib.tensorflow.as_tensorflow_layer(self.grad_reg_op, 'Regulariser')

        # placeholders for forward model
        self.x_ini = tf.placeholder(shape=[None, self.pic_size, self.pic_size, 1], dtype=tf.float32, name='InitialGuess')
        self.x_true = tf.placeholder(shape=[None, self.pic_size, self.pic_size, 1], dtype=tf.float32, name='GroundTruth')
        self.y = tf.placeholder(shape=[None, self.ray_transf.range.shape[0], self.ray_transf.range.shape[1], 1], dtype=tf.float32,
                           name='Measurement_Data')

        # set up the forward model
        x = self.x_ini
        self.weights_recon = self.get_weights()
        for i in range(self.iterations):
            # calculate the gradient of the data error
            with tf.name_scope('Data_gradient'):
                measurement = tf.exp(-self.attenuation_coeff * self.tf_ray(x))
                g_x = self.attenuation_coeff * self.tf_ray_adj(self.y - measurement)
                tf.summary.scalar('Data_gradient_Norm', tf.norm(g_x))
                g_reg = self.tf_reg(x)
                tf.summary.scalar('Regulariser_gradient_Norm', tf.norm(g_reg))

                # use the network model defined in
                x_update = self.forward_model(x, g_x, g_reg, self.weights_recon)

                tf.summary.scalar('x_update', tf.norm(x_update))
                x = x + x_update
        self.result = x


        # define L2 loss function
        with tf.name_scope('L2-Loss'):
            self.lossL2 = tf.reduce_mean(tf.reduce_sum((self.result - self.x_true) ** 2, axis=(1, 2)))

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Optimizer for L2 loss
        with tf.name_scope('L2-optimizer'):
            self.optimizer_L2 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.lossL2, global_step=self.global_step,
                                                                       var_list=self.weights_recon)

        # finish setup. Should always be executed unless this init is called in an init of a subclass
        if final:
            self.finish_setup()



    def finish_setup(self):
        # merge training log
        self.merged = tf.summary.merge_all()

        # training logger
        self.writer = tf.summary.FileWriter('Data/' + self.model_name + '/logs', self.sess.graph)

        # Initialize Saver
        self.saver = tf.train.Saver()

        # initialize variables
        tf.global_variables_initializer().run()

        #restore latest save
        self.load()

    # loads the latest savepoint
    def load(self):
        if os.listdir('Data/' + self.model_name + '/Saves/'):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join('Data', self.model_name, 'Saves', '')))
            print('Save restored')
        else:
            print('No save found')

    # Saves the current model
    def save(self):
        self.saver.save(self.sess, 'Data/' + self.model_name + '/Saves/model', global_step=self.global_step)
        print('Progress saved')

    # set up function to generate simulted MRI measurements form training data
    def simulated_measurements(self, batch, validation_data = False):
        x_ini = np.empty(shape=[batch, self.pic_size, self.pic_size, 1])
        x_true = np.empty(shape=[batch, self.pic_size, self.pic_size, 1])
        y = np.empty(shape=[batch, self.ray_transf.range.shape[0], self.ray_transf.range.shape[1], 1])
        label = np.empty(shape=[batch])

        for i in range(batch):
            if not validation_data:
                rand = random.randint(0, self.number_training_data-1)
                pic_original = self.train_data[rand,:]
                pic_reshaped = np.reshape(pic_original, [self.pic_size, self.pic_size])
                label[i] = self.train_labels[rand]
            else:
                rand = random.randint(0, self.number_eval_data-1)
                pic_original = self.eval_data[rand,:]
                pic_reshaped = np.reshape(pic_original, [self.pic_size, self.pic_size])
                label[i] = self.eval_labels[rand]

            # Wrap the picture inside an ODL 'space' element
            odl_pic = self.space.element(pic_reshaped)
            measure = np.exp(-self.attenuation_coeff * self.ray_transf(odl_pic))

            # Add poisson noise
            noisy_data = odl.phantom.poisson_noise(measure * self.photons_per_unit) / self.photons_per_unit
            initial_guess = self.fbp(-np.log(noisy_data + (1/self.photons_per_unit)) / self.attenuation_coeff)

            # fill in the output data
            x_ini[i,:,:,0] = initial_guess
            x_true[i,:,:,0] = pic_reshaped
            y[i,:,:,0] = noisy_data#
            # y[i,:,:,0] = measure
            # print('noisy minimum: ' + str(np.min(y)))
            # print('original minimum: ' + str(np.min(measure)))
        return x_ini, x_true, y, label

    # method to compute the TV regularisation for comparison
    def tv_reconsruction(self, y, param = 1000000):
        # the operators
        gradients = odl.Gradient(self.space, method='forward')
        operator = odl.BroadcastOperator(self.ray_transf, gradients)
        # define empty functional to fit the chambolle_pock framework
        g = odl.solvers.ZeroFunctional(operator.domain)

        # compute transformed data
        # ensure y stays away from 0
        y_cut = np.maximum(y, 0.03)
        data = -(np.log(y_cut))/self.attenuation_coeff

        # the norms
        l1_norm = param * odl.solvers.L1Norm(gradients.range)
        l2_norm_squared = odl.solvers.L2NormSquared(self.ray_transf.range).translated(data)
        functional = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)

        # Find parameters
        op_norm = 1.1 * odl.power_method_opnorm(operator)
        tau = 10.0 / op_norm
        sigma = 0.1 / op_norm
        niter = 5000

        # find starting point
        x = self.fbp(data)

        # Run the optimization algoritm
        odl.solvers.chambolle_pock_solver(x, functional, g, operator, tau = tau, sigma = sigma, niter=niter)

        # plot results
        plt.figure(1)
        plt.imshow(x)
        plt.show()

    # compute gradients
    def data_reg_gradients(self,x,y):
        x_input = x[0,:,:,0]
        y_input= y[0,:,:,0]
        measurement = np.exp(-self.attenuation_coeff * self.ray_transf(x_input))
        g_x_raw = self.attenuation_coeff * self.ray_transf.adjoint(y_input - measurement)
        g_reg_raw = self.grad_reg_op(x)
        g_x = np.empty(shape=[1,28,28,1])
        g_reg = np.empty(shape=[1,28,28,1])
        g_x[0,:,:,0] = g_x_raw
        g_reg[0,:,:,0] = g_reg_raw
        return g_x, g_reg

    # saves the current picture
    def save_pic(self, true, fbp, alg, iteration):
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
        plot_dir = 'Data/'+ self.model_name + '/Pics/' + 'Iteration_' + str(iteration) + '.png'
        plt.savefig(plot_dir)

    # methode to evaluate model performance while training:
    def evaluate(self):
        x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(100, validation_data=True)
        summary, step_number,  picture, loss_evaluation = self.sess.run(
            [self.merged, self.global_step, self.result, self.lossL2],
            feed_dict={self.x_ini: x_ini_np, self.x_true: x_true_np,
                       self.y: y_np})
        original_loss = np.square(x_ini_np - x_true_np)
        print('Iteration: ' + str(step_number) + ', Loss: ' +
              "{0:.6g}".format(loss_evaluation) + ', Original Loss: ' + "{0:.6g}".format(
            (original_loss.sum()) / 100) +
              ', Net Improvement: ' + str(((original_loss.sum()) / 100) - loss_evaluation))
        self.writer.add_summary(summary, step_number)
        self.save_pic(x_true_np, x_ini_np, picture, step_number)

    # the training routine
    def train_L2(self, training_steps):
        for i in range(training_steps):
            #model evaluation
            if i %  10 == 0:
                self.evaluate()
            #train the network
            x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(self.batch_size, validation_data=False)
            self.sess.run(self.optimizer_L2, feed_dict={self.x_ini: x_ini_np, self.x_true: x_true_np,
                                                        self.y: y_np})
            self.save()


class JointTraining(l2):
    weightL2_combinedNorms = 0
    model_name = 'Classification_Loss'

    def load_default_classifier(self):
        classifier_res = tf.train.Saver(var_list=self.weights_classifier)
        classifier_res.restore(self.sess, tf.train.latest_checkpoint('classifier/weights/'))
        print('Restored pretrained classifier weights')

    def load(self):
        if os.listdir('Data/' + self.model_name + '/Saves/'):
            self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join('Data', self.model_name, 'Saves', '')))
            print('Save restored')
        else:
            self.load_default_classifier()

    def get_classifier_weights(self):
        return ut.classifier_variables(True)

    def classifier_model(self, input, weights):
        return ut.classifier_model(input, weights, 0.0)

    def __init__(self):
        super(JointTraining, self).__init__(final=False)

        self.labels = tf.placeholder(shape=[None], dtype=tf.float32, name='CorrectLabels')
        self.ohl = tf.one_hot(tf.cast(self.labels, tf.int32), depth=10)
        # extend forward model by classifier
        self.weights_classifier = self.get_classifier_weights()
        self.clas_result = self.classifier_model(self.result, self.weights_classifier)

        # define classification evaluation
        with tf.name_scope('Evaluierer'):
            predictions_MN = tf.argmax(input=self.clas_result, axis=1)
            correct_predictions = tf.equal(predictions_MN, tf.cast(self.labels, tf.int64))
            self.eval_metric = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # define classification loss function
        with tf.name_scope('classification-Loss'):
            self.lossClas = tf.losses.softmax_cross_entropy(onehot_labels=self.ohl, logits=self.clas_result)

        # define linear combination of L2 and classification loss
        with tf.name_scope('Loss_sum_class_l2'):
            self.lossL2Clas = self.weightL2_combinedNorms * self.lossL2 + self.lossClas

        # Optimizer for Joint Training
        with tf.name_scope('Joint_learning'):
            self.optimizer_joint = tf.train.AdamOptimizer(self.learning_rate).minimize(self.lossL2Clas,
                                                                                   global_step=self.global_step,
                                                                                   var_list=self.weights_recon + self.weights_classifier)
        # Optimizer for classifcation loss
        with tf.name_scope('Learning_clssifier_loss'):
            self.optimizer_class_loss = tf.train.AdamOptimizer(self.learning_rate).minimize(self.lossL2Clas,
                                                                                                global_step=self.global_step,
                                                                                            var_list=self.weights_recon)
        self.finish_setup()

    # change inherited methode to include CE and classification acc
    def evaluate(self):
        x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(100, validation_data=True)
        summary, step_number,  picture, loss_evaluation, CE, acc = self.sess.run(
            [self.merged, self.global_step, self.result, self.lossL2, self.lossClas, self.eval_metric],
            feed_dict={self.x_ini: x_ini_np, self.x_true: x_true_np,
                       self.y: y_np, self.labels: lab_np})
        original_loss = np.square(x_ini_np - x_true_np)
        print('Iteration: ' + str(step_number) + ', Loss: ' +
              "{0:.6g}".format(loss_evaluation) + ', Original Loss: ' + "{0:.6g}".format(
            (original_loss.sum()) / 100) +
              ', Net Improvement: ' + str(((original_loss.sum()) / 100) - loss_evaluation) + ', CE: '
              + str(CE) + ', Accuracy: ' + str(acc))
        self.writer.add_summary(summary, step_number)
        self.save_pic(x_true_np, x_ini_np, picture, step_number)

    #### to be written!
    def train_L2(self, training_steps):
        for i in range(training_steps):
            #model evaluation
            if i %  10 == 0:
                self.evaluate()
            #train the network
            x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(self.batch_size, validation_data=False)
            self.sess.run(self.optimizer_L2, feed_dict={self.x_ini: x_ini_np, self.x_true: x_true_np,
                                                        self.y: y_np, self.labels: lab_np})
        self.save()

    def train_class_loss(self, training_steps):
        for i in range(training_steps):
            #model evaluation
            if i %  10 == 0:
                self.evaluate()
            #train the network
            x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(self.batch_size, validation_data=False)
            self.sess.run(self.optimizer_class_loss, feed_dict={self.x_ini: x_ini_np, self.x_true: x_true_np,
                                                        self.y: y_np, self.labels: lab_np})
        self.save()

    def train_jointly(self, training_steps):
        for i in range(training_steps):
            #model evaluation
            if i %  10 == 0:
                self.evaluate()
            #train the network
            x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(self.batch_size, validation_data=False)
            self.sess.run(self.optimizer_joint, feed_dict={self.x_ini: x_ini_np, self.x_true: x_true_np,
                                                        self.y: y_np, self.labels: lab_np})
        self.save()


class Adverserial(l2):
    model_name = 'Adverserial_Loss'
    learning_rate_adv = 0.001
    weightL2_combinedNorms = 0.15

    def get_adverserial_weights(self):
        return ut.adversarial_weights(self.model_name)

    def adverserial_model(self, input, weights):
        return ut.adverserial_network(input, weights, '')

    def __init__(self):
        super(Adverserial, self).__init__(final=False)

        # The network for adverserial training
        self.adv_weights = self.get_adverserial_weights()
        scaling_factor = 1
        self.adv_class_net = self.adverserial_model(scaling_factor*self.result, self.adv_weights)
        self.adv_class_true = self.adverserial_model(scaling_factor*self.x_true, self.adv_weights)

        # loss of adverserial Network during training given by misclassification loss of true data and network data
        self.loss_adv = -tf.reduce_mean(tf.log(self.adv_class_true) + tf.log(1. - self.adv_class_net))

        # evaluation metric for classification
        self.acc_adv = (tf.reduce_mean(tf.cast(tf.greater(0.5, self.adv_class_net), tf.float32)) +
                        tf.reduce_mean(tf.cast(tf.greater(self.adv_class_true, 0.5), tf.float32))) / 2

        # loss of the generator trying to fool the adverserial network
        self.loss_gen = -tf.reduce_mean(tf.log(self.adv_class_net))

        # evaluation metric for generator
        self.acc_gen = tf.reduce_mean(tf.cast(tf.greater(self.adv_class_net, 0.5), tf.float32))

        # the optimizers
        self.optimizer_adverserial = tf.train.AdamOptimizer(self.learning_rate_adv).minimize(self.loss_adv,
                                                                                     var_list=self.adv_weights)
        self.optimizer_generator = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_gen + self.lossL2 * self.weightL2_combinedNorms,
            var_list=self.weights_recon,
            global_step=self.global_step)


        self.finish_setup()

    def evaluate(self):
        x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(100, validation_data=True)
        summary, step, pic, accuracy, crossE, l2loss = self.sess.run([self.merged, self.global_step, self.result,
                                                             self.acc_gen, self.loss_gen, self.lossL2],
                                                       feed_dict={self.x_ini: x_ini_np, self.x_true: x_true_np,
                                                                  self.y: y_np})
        print('Iteration:' + str(step) + ', Generator fooling perc.: ' + str(accuracy) +
              ', CE: ' + str(crossE) + ' , L2-Loss: ' + str(l2loss))
        self.writer.add_summary(summary, step)
        self.save_pic(x_true_np, x_ini_np, pic, step)

    def evaluate_adv_net(self):
        x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(100, validation_data=True)
        out_true, out_recon, accuracy, crossE = self.sess.run([self.adv_class_true, self.adv_class_net, self.acc_adv, self.loss_adv], feed_dict={self.x_ini: x_ini_np,
                                                                                   self.x_true: x_true_np, self.y: y_np})
        print('Discrimination accuracy: ' + str(accuracy) +', CE: ' + str(crossE) + ', Output Net True: '
              + str(out_true[0,0]) + ', Output Net Fake: ' + str(out_recon[0,0]))


    def train_adv(self, training_steps, steps_gen, steps_adv):
        # actual adverserial training
        for j in range(training_steps):
            # train the adverserial network to discriminate between real data and network data
            print('Training Adverserial Network')
            for k in range(steps_adv):
                self.evaluate_adv_net()
                x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(self.batch_size, validation_data=False)
                self.sess.run(self.optimizer_adverserial, feed_dict={self.x_ini: x_ini_np,
                                                                     self.x_true: x_true_np, self.y: y_np})
            print('Training Generator')
            for k in range(steps_gen):
                x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(self.batch_size, validation_data=False)
                self.sess.run(self.optimizer_generator, feed_dict={self.x_ini: x_ini_np,
                                                                   self.x_true: x_true_np, self.y: y_np})
            self.evaluate()
        self.save()


