import ImageRecon_classes as ir

class Loss_Class(ir.Classification_Loss):
    model_name = 'ClassificationLoss'

class Loss_Jointly(ir.Classification_Loss):
    model_name = 'JointlyTrained'

class Train_Classifier_Only(ir.Classification_Loss):
    model_name = 'TrainClassifierOnly'


if __name__ == '__main__':

    training_steps = 1000
    if 1:
        recon = Loss_Class()
        recon.train_class_loss(training_steps=training_steps)
        recon.end()

    if 1:
        recon2 = Loss_Jointly()
        recon2.train_jointly(training_steps)
        recon2.end()

    if 1:
        recon3 = Train_Classifier_Only()
        recon3.train_classifier_only(training_steps)
        recon3.end()