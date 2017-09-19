import ImageRecon_classes as ir

class Loss_L2(ir.l2):
    model_name = 'L2-Loss'

if __name__ == '__main__':
    recon = Loss_L2()
    recon.train_L2(3000)