import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from Optimizer import Joint_Optimizer
from params import *
from Image import Image
from timeit import default_timer as timer
from Contour_App import Contour_App
from utils import *

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.use("TKAgg")

if __name__ == '__main__':
    contour_app = Contour_App()
    u0, image0, gt = contour_app.run()

    image = Image(image0, ground_truth = gt, noise_std = 0.15)
    optimizer = Joint_Optimizer(cow_params_seg, cow_params_recon, u0, image, iterations = 3, verbose = True, plotting = False)
    
    start = timer()
    u, u0, im, dice_score = optimizer.run()
    end = timer()

    print(f'Runtime: {end - start}')      
    print(f'DICE-score: {dice_score}')
    print(f'Initial PSNR: {PSNR(image.y, image0)}')
    print(f'Final PSNR: {PSNR(im, image0)}')

    # -------------- Plot inital segmentation -------------- 
    # Create mask for plotting transparent region
    M, N = u.shape
    mask = np.ones_like(u) - u    
    opaque_layer = np.zeros([M, N, 4])
    opaque_layer[:,:,0:3] = 0
    opaque_layer[:,:,3] = mask

    fig1 = plt.figure()
    plt.imshow(image.y)
    plt.axis('off')
    plt.title('Observation', fontweight = 'bold', fontsize = 18)
    plt.savefig('./Results/cow_observation.eps', bbox_inches='tight')

    fig2 = plt.figure()
    plt.imshow(image0)
    plt.axis('off')
    plt.imshow(opaque_layer, alpha = 0.65)
    plt.title('Final Segmentation', fontweight = 'bold', fontsize = 18)
    plt.savefig('./Results/cow_seg_joint.eps', bbox_inches='tight')

    fig3 = plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.title('Final Reconstruction', fontweight = 'bold', fontsize = 18)
    plt.savefig('./Results/cow_recon_joint.eps', bbox_inches='tight')