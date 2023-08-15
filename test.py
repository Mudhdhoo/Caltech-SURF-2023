import matplotlib.pyplot as plt
from Optimizer import Joint_Optimizer
from params import *
from Image import Image

if __name__ == '__main__':
    image = Image('heart')
    optimizer = Joint_Optimizer(heart_params_seg, recon_params, image, iterations = 3, verbose = True)
    u, im = optimizer.run()
    plt.imshow(u)
    plt.show()