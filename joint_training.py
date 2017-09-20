import ImageRecon_classes as ir
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class Loss_L2(ir.Classification_Loss):
    model_name = 'L2LossClassificationComparison'

class Loss_Class(ir.Classification_Loss):
    model_name = 'ClassificationLoss'

class Loss_Jointly(ir.Classification_Loss):
    model_name = 'JointlyTrained'

class Train_Classifier_Only(ir.Classification_Loss):
    model_name = 'TrainClassifierOnly'

def compare_class_pref():
    # created needed folder
    if not os.path.exists('Data/Evaluations/Comparison_Classification'):
        try:
            os.makedirs('Data/Evaluations/Comparison_Classification')
        except OSError:
            pass
        print('Evaluation directory created')
    l2Loss = np.zeros(shape=[4])
    CE = np.zeros(shape=[4])
    acc = np.zeros(shape=[4])
    pics = np.zeros(shape=[4, 28, 28])
    rec = Loss_L2()
    x_ini, x_true, y, label = rec.simulated_measurements(20, validation_data=True)
    image, l2Loss[0], CE[0], acc[0] = rec.full_model_evaluation(x_ini, x_true, y, label)
    pics[0,...] = image[0,...,0]
    rec.end()
    rec = Loss_Class()
    image, l2Loss[1], CE[1], acc[1] = rec.full_model_evaluation(x_ini, x_true, y, label)
    pics[1,...] = image[0,...,0]
    rec.end()
    rec = Loss_Jointly()
    image, l2Loss[3], CE[3], acc[3] = rec.full_model_evaluation(x_ini, x_true, y, label)
    pics[3,...] = image[0,...,0]
    rec.end()
    rec = Train_Classifier_Only()
    image, l2Loss[2], CE[2], acc[2] = rec.full_model_evaluation(x_ini, x_true, y, label)
    pics[2,...] = image[0,...,0]
    rec.end()

    plt.figure(1)
    plt.subplot(4,4,1)
    plt.imshow(pics[0, :, :])
    plt.title('L2-trained')
    plt.subplot(4,4,2)
    plt.imshow(pics[1, :, :])
    plt.title('Class. Loss')
    plt.subplot(4,4,3)
    plt.imshow(pics[2, :, :])
    plt.title('Class. Trained Only')
    plt.subplot(4,4,4)
    plt.imshow(pics[3, :, :])
    plt.title('Jointly trained')
    plt.savefig('Data/Evaluations/Comparison_Classification')






if __name__ == '__main__':

    compare_class_pref()

    training_steps = 1000
    if 0:
        recon = Loss_Class()
        recon.train_class_loss(training_steps=training_steps)
        recon.end()

    if 0:
        recon2 = Loss_Jointly()
        recon2.train_jointly(training_steps)
        recon2.end()

    if 0:
        recon3 = Train_Classifier_Only()
        recon3.train_classifier_only(training_steps)
        recon3.end()

    if 0:
        recon4 = Loss_L2()
        recon4.train_L2(training_steps)
        recon4.end()