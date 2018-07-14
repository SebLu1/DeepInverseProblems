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

class C1(ir.Classification_Loss):
    model_name = 'JointlyTrained_C0.0001'
    weightL2_combinedNorms = 0.0001

class C2(ir.Classification_Loss):
    model_name = 'JointlyTrained_C0.001'
    weightL2_combinedNorms = 0.001

class C3(ir.Classification_Loss):
    model_name = 'JointlyTrained_C0.01'
    weightL2_combinedNorms = 0.01

class C4(ir.Classification_Loss):
    model_name = 'JointlyTrained_C0.1'
    weightL2_combinedNorms = 0.1

class C5(ir.Classification_Loss):
    model_name = 'JointlyTrained_C1.0'
    weightL2_combinedNorms = 1.0


if __name__ == '__main__':
    training_steps = 1500
    # pretraining the models for L2 loss
    if 0:
        recon = Loss_Class()
        recon.train_L2(training_steps)
        recon.end()

        recon2 = Loss_Jointly()
        recon2.train_L2(training_steps)
        recon2.end()

        recon3 = Train_Classifier_Only()
        recon3.train_L2(training_steps)
        recon3.end()

        recon4 = Loss_L2()
        recon4.train_L2(training_steps)
        recon4.end()

    training_steps = 5000
    if 0:
        recon = Loss_Class()
        recon.train_class_loss(training_steps)
        recon.end()

        recon2 = Loss_Jointly()
        recon2.train_jointly(training_steps)
        recon2.end()

        recon3 = Train_Classifier_Only()
        recon3.train_classifier_only(training_steps)
        recon3.end()

        recon4 = Loss_L2()
        recon4.train_L2(training_steps)
        recon4.end()

    # visualizations
    if 0:
        recon = Loss_Class()
        recon.ozan_vis(5)
        recon.end()

        recon = Loss_Jointly()
        recon.ozan_vis(5)
        recon.end()

        recon = Train_Classifier_Only()
        recon.ozan_vis(5)
        recon.end()

        recon = Loss_L2()
        recon.ozan_vis(5)
        recon.end()

    # run this for a bunch of different values of the weighting parameter C
    C_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
    if 0:
        for C in C_list:
            class JointTrainingWithC(ir.Classification_Loss):
                model_name = 'JointlyTrained_C' + str(C)
                weightL2_combinedNorms = C

            print('C : ' + str(C))
            recon = JointTrainingWithC()
            recon.train_L2(4000)
            print('C : ' + str(C))
            recon.train_jointly(8000)
            recon.end()
            print('C : ' + str(C))






    #compare_class_perf()