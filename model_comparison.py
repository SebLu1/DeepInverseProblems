import joint_training as jt
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


def visualize_models(number, model_list):
    path = 'Data/Evaluations/Visual_Comparison/Sample' + str(number)
    create_dir(path)

    # plot original image and fbp plus corresponding probablities and get data needed.
    recon = jt.Loss_L2()
    x_ini, x_true, y, label = recon.simulated_measurements(1, validation_data=True)
    labels, output_pic, output_labels, fbp_clas = recon.visual_model_evaluation(x_ini, x_true, y, label)

    true_labels = []
    for k in range(len(labels[0])):
        true_labels.append([labels[0][k]])
    fbp_labels = []
    for k in range(len(fbp_clas[0])):
        fbp_labels.append([fbp_clas[0][k]])
    columns = ('Probability')
    rowLabels = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # true figure
    plt.figure()
    plt.imshow(x_true[0, ..., 0], cmap='gray')
    plt.axis('off')
    # Add a table at the bottom of the axes
    plt.table(cellText=true_labels,
              rowLabels=rowLabels,
              colLabels=columns,
              loc='bottom')
    plt.savefig(path+'/TrueImage.png', bbox_inches='tight')
    plt.close()

    # fbp figure
    plt.figure()
    plt.imshow(x_ini[0, ..., 0], cmap='gray')
    plt.axis('off')
    # Add a table at the bottom of the axes
    plt.table(cellText=fbp_labels,
              rowLabels=rowLabels,
              colLabels=columns,
              loc='bottom')
    plt.savefig(path+'/FBP.png', bbox_inches='tight')
    plt.close()
    recon.end()

    # plot the fucking sinogram
    sinogram = -np.log(y[0,...,0] + (5.0 / recon.photons_per_unit)) / recon.attenuation_coeff
    plt.figure()
    plt.imshow(sinogram, cmap='gray')
    plt.savefig(path+'/Sinogram.png')
    plt.close()

    # plot results and model class distributions for all classes in classlist
    for model in model_list:
        recon = model()
        name = recon.model_name
        labels, output_pic, output_labels, fbp_clas = recon.visual_model_evaluation(x_ini, x_true, y, label)

        # make the labels
        out_labels = []
        for k in range(len(output_labels[0])):
            out_labels.append([output_labels[0][k]])

        plt.figure(1)
        plt.imshow(output_pic[0, ..., 0], cmap='gray')
        plt.axis('off')
        # Add a table at the bottom of the axes
        plt.table(cellText=out_labels,
                  rowLabels=rowLabels,
                  colLabels=columns,
                  loc='bottom')
        plt.savefig(path + '/{}.jpg'.format(name), bbox_inches='tight')
        plt.close()
        recon.end()

def compare_models(model_list, batch_size = 2000):
    path = 'Data/Evaluations/Comparison_Classification'
    create_dir(path)

    data = jt.Loss_L2()
    x_ini, x_true, y, label = data.simulated_measurements(batch_size, validation_data=True)
    data.end()

    CE = {}
    L2 = {}
    acc = {}
    tl = {}

    for model in model_list:
        recon = model()
        name = recon.model_name
        image, l2Loss, crossEntro, accuracy, total_loss = recon.full_model_evaluation(x_ini, x_true, y, label)
        recon.end()

        CE[name] = crossEntro
        L2[name] = l2Loss
        acc[name] = accuracy
        tl[name] = total_loss

        print('{}. CE: {}, acc: {}, L2: {}'.format(name, crossEntro, accuracy, l2Loss))

    return CE, L2, acc, tl


model_list = [jt.Loss_L2, jt.Loss_Class, jt.Loss_Jointly, jt.Train_Classifier_Only, jt.C1, jt.C2, jt.C3, jt.C4, jt.C5]

for k in range(15):
    visualize_models(k, model_list)

CE, L2, acc, tl = compare_models(model_list = model_list, batch_size=4000)
print('CE')
print(CE)
print('acc')
print(acc)
print('L2')
print(L2)