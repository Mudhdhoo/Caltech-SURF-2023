import matplotlib.pyplot as plt
import matplotlib
import os
from Optimizer import Joint_Optimizer
from params import *
from Image import Image
from timeit import default_timer as timer
from Contour_App import Contour_App

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.use("TKAgg")

if __name__ == '__main__':
    contour_app = Contour_App()
    u0, image, gt = contour_app.run()

    image = Image(image, ground_truth = gt, noise_std = 0.15)
    optimizer = Joint_Optimizer(cow_params_seg, cow_params_recon, u0, image, iterations = 2, verbose = True, plotting = True)
    
    start = timer()
    u, im = optimizer.run()
    end = timer()

    print(f'Runtime: {end - start}')      
    plt.imshow(u)
    plt.title('Final Segmentation')
    plt.show()
