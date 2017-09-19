import ImageRecon_classes as ir

class Loss_Class(ir.Classification_Loss):
    model_name = 'ClassificationLoss'

class Loss_Jointly(ir.Classification_Loss):
    model_name = 'JointlyTrained'

class Train_Classifier_Only(ir.Classification_Loss):
    model_name = 'TrainClassifierOnly'


if __name__ == '__main__':

    if 1:
        recon = Loss_Class()
        recon.train_L2(1500)
        recon.end()

    if 0:
        recon = Loss_Class()
        recon.train_class_loss(2000)
        recon.end()

        recon2 = Loss_Jointly()
        recon2.train_jointly(2000)
        recon2.end()

        recon3 = Loss_Jointly()
        recon3.train_classifier_only(2000)
        recon3.end()