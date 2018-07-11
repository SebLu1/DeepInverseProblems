import L2
import joint_training as jt
import adverserial as ad
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass
        print(path+ ' created')


def visualize_models(number):
    path  = 'Data/Evaluations/Visual_Comparison' + number
    create_dir(path)

    rec = L2.Loss_L2()
    x_ini, x_true, y, label = rec.simulated_measurements(1, validation_data=True)
    image = rec.compute_data(x_ini, x_true, y, label)
    rec.end()
    plt.figure(1)
    plt.imshow(image[0,...])
    plt.axis('off')
    plt.title('L2')


    rec = jt.Loss_Class()
    image = rec.compute_data(x_ini, x_true, y, label)
    pics[1,...] = image[0,...,0]
    rec.end()
    rec = jt.Loss_Jointly()
    image = rec.compute_data(x_ini, x_true, y, label)
    pics[2,...] = image[0,...,0]
    rec.end()

    plt.figure()
    plt.subplot(131)
    plt.imshow(pics[0,...])
    plt.axis('off')
    plt.title('L2')
    plt.subplot(132)
    plt.imshow(pics[1,...])
    plt.axis('off')
    plt.title('Class. Loss')
    plt.subplot(133)
    plt.imshow(pics[2,...])
    plt.axis('off')
    plt.title('Joint Training')
    plt.savefig('Data/Evaluations/Visual_Comparison/result')
    plt.close()

def visualize_Ozan():
    x_ini_np, x_true_np, y_np, lab_np = self.simulated_measurements(1)
    labels, output_pic, output_labels, fbp_clas = self.sess.run(
        [self.ohl, self.result, self.probabilities, self.fbp_probabilities],
        feed_dict={self.x_ini: x_ini_np, self.x_true: x_true_np,
                   self.y: y_np, self.labels: lab_np})
    true_labels = []
    for k in range(len(labels[0])):
        true_labels.append([labels[0][k]])
    recon_labels = []
    for k in range(len(output_labels[0])):
        recon_labels.append([output_labels[0][k]])
    fbp_labels = []
    for k in range(len(fbp_clas[0])):
        fbp_labels.append([fbp_clas[0][k]])
    columns = ('Probability')
    rowLabels = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # true figure
    plt.figure(1)
    plt.imshow(x_true_np[0, ..., 0], cmap='gray')
    plt.axis('off')
    # Add a table at the bottom of the axes
    plt.table(cellText=true_labels,
              rowLabels=rowLabels,
              colLabels=columns,
              loc='bottom')
    plt.savefig('Data/Evaluations/' + self.model_name + '_True_' + str(i) + '.png', bbox_inches='tight')
    plt.close()

    # reconstructed figure
    plt.figure(2)
    plt.imshow(output_pic[0, ..., 0], cmap='gray')
    plt.axis('off')
    # Add a table at the bottom of the axes
    plt.table(cellText=recon_labels,
              rowLabels=rowLabels,
              colLabels=columns,
              loc='bottom')
    plt.savefig('Data/Evaluations/' + self.model_name + '_Reconstruction_' + str(i) + '.png', bbox_inches='tight')
    plt.close()

    # fbp figure
    plt.figure(3)
    plt.imshow(x_ini_np[0, ..., 0], cmap='gray')
    plt.axis('off')
    # Add a table at the bottom of the axes
    plt.table(cellText=fbp_labels,
              rowLabels=rowLabels,
              colLabels=columns,
              loc='bottom')
    plt.savefig('Data/Evaluations/' + self.model_name + '_FBP_' + str(i) + '.png', bbox_inches='tight')
    plt.close()

visualize_models()