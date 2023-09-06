import matplotlib.pyplot as plt
import os
from Optimizer import Joint_Optimizer
from params import *
from Image import Image
from scipy.io import loadmat
from timeit import default_timer as timer
plt.rcParams["font.family"] = "Times New Roman"

if __name__ == '__main__':
    gt = loadmat(os.path.join('images','heart_truth'))['groundtruth']
    image = Image('cow', ground_truth = None, scale = 1, noise_std = 0.1)
    optimizer = Joint_Optimizer(cow_params_seg, cow_params_recon, image, iterations = 6, verbose = True, plotting = True)
    
    start = timer()
    u, im = optimizer.run()
    end = timer()

    print(f'Runtime: {end - start}')

    plt.imshow(u)
    plt.title('Final Segmentation')
    plt.show()