import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from Optimizer import Joint_Optimizer
from params import *
from Image import Image
from Contour_App import Contour_App

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.use("TKAgg")

if __name__ == '__main__':
    contour_app = Contour_App()     
    u0, image0, gt = contour_app.run()      # Run hand-drawing GUI
    
    image = Image(image0, ground_truth = gt, noise_std = 0.15)      # Create image instance

    optimizer = Joint_Optimizer(heart_params_seg, heart_params_recon, u0, image, iterations = 3, verbose = True, plotting = True)      # Create optimizer instance
    
    u, u0, im, dice = optimizer.run()   # Run optimizer

    M, N = u.shape
    mask = np.ones_like(u) - u    
    opaque_layer = np.zeros([M, N, 4])
    opaque_layer[:,:,0:3] = 0
    opaque_layer[:,:,3] = mask    
    plt.imshow(image0)
    plt.imshow(opaque_layer)
    plt.title('Final Segmentation')
    plt.show()


