import ImageRecon_classes as ir

class Loss_adverserial(ir.Adverserial):
    model_name = 'Adverserial_Loss'

if __name__ == '__main__':
    recon = Loss_adverserial()
    recon.train_L2(500)
    recon.train_adv(5,0,10)
    recon.train_adv(200, 3, 7)