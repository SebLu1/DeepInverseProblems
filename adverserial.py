import ImageRecon_classes as ir

class Loss_adverserial(ir.Adverserial):
    ## specifies the weight factor of L2 loss in comparison to classification loss
    weightL2_combinedNorms = 0.4
    model_name = 'Adverserial_Loss'

if __name__ == '__main__':
    recon = Loss_adverserial()
    recon.train_adv(400, 5, 5)