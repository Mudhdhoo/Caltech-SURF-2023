import matplotlib.pyplot as plt
import matplotlib
import os
from Optimizer import Joint_Optimizer
from params import *
from Image import Image
from scipy.io import loadmat
from timeit import default_timer as timer
from Contour_App import Contour_App

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.use("TKAgg")

if __name__ == '__main__':
    gt_heart = loadmat(os.path.join('images','heart_truth'))['groundtruth']
    contour_app = Contour_App()
    u0, image = contour_app.run()

    image = Image(image, ground_truth = None, noise_std = 0.15)
    optimizer = Joint_Optimizer(heart_params_seg, heart_params_recon, u0, image, iterations = 3, verbose = True, plotting = True)
    
    start = timer()
    u, im = optimizer.run()
    end = timer()

    print(f'Runtime: {end - start}')      
    plt.imshow(u)
    plt.title('Final Segmentation')
    plt.show()
