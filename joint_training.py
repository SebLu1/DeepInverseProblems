import ImageRecon_classes as ir

class Loss_Class(ir.Classification_Loss):
    model_name = 'ClassificationLoss'

class Loss_Jointly(ir.Classification_Loss):
    model_name = 'JointlyTrained'


if __name__ == '__main__':
    recon = Loss_Class()
    recon.train_L2(1000)
    recon.train_class_loss(2000)
    recon.end()

    recon2 = Loss_Jointly()
    recon2.train_L2(1000)
    recon2.train_jointly(2000)
    recon2.end()