import L2
import joint_training as jt
import adverserial as ad
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



def visualize_models():
    if not os.path.exists('Data/Evaluations/Visual_Comparison'):
        try:
            os.makedirs('Data/Evaluations/Visual_Comparison')
        except OSError:
            pass
        print('Visualization directory created')

    pics = np.zeros(shape=[4, 28, 28])
    rec = L2.Loss_L2()
    x_ini, x_true, y, label = rec.simulated_measurements(1, validation_data=True)
    image = rec.compute_data(x_ini, x_true, y, label)
    pics[0,...] = image[0,...,0]
    rec.end()
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

visualize_models()