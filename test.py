import matplotlib.pyplot as plt
import os
from Optimizer import Joint_Optimizer
from params import *
from Image import Image
from scipy.io import loadmat

if __name__ == '__main__':
    gt = loadmat(os.path.join('images','heart_truth'))['groundtruth']
    image = Image('heart', ground_truth = gt)
    optimizer = Joint_Optimizer(heart_params_seg, recon_params, image, iterations = 3, verbose = True, plotting = True)
    u, im = optimizer.run()
    plt.imshow(u)
    plt.show()