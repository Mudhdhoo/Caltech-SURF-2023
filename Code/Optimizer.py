import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Segmenter import Segmenter
from Reconstructor import Reconstructor
from Contour_App import Contour_App
from Image import Image
from Parameters import *
from params import *
from timeit import default_timer as timer

class Joint_Optimizer:
    """
    Contains the main joint reconstruction-segmentation loop.

    ------------ Parameters ------------
    seg_params: Segmentation_Params
        Dataclass for storing the parameters for the segmentation.
    
    recon_params: Reconstruction_Params
        Dataclass for storing the parameters for the reconstruction.

    image: Image
        Image class containing the image to be optimized.

    iterations: int
        Number of iterations in the main optimization loop.

    verbose: Bool
        True for information output, False otherwise.
    """

    def __init__(self, seg_params:Segmentation_Params, recon_params:Reconstruction_Params, u0, image, iterations:int, verbose = False, plotting = True) -> None:
        self.segmenter = Segmenter(seg_params)
        self.reconstructor = Reconstructor(recon_params, 'TV', TV_weight = 1)
        self.contour_app = Contour_App()
        self.iterations = iterations
        self.verbose = verbose
        self.plotting = plotting
        self.u0 = u0
        self.image = image

    def run(self):
        """
        compute cheap reconstruction
        compute initial contour
        for iteration in iterations do:
            compute segmentation
            compute reconstruction based on segmentaion
            update image with new reconstruction
        """
        start = timer()
        # Pre-processing cheap reconstruction
        y = self.image.image
        Im0 = self.reconstructor.cheap_reconstruction(y)
        self.image.update_image(Im0)

        # Main optimization loop
        u = self.u0
        for iteration in range(0, self.iterations):
            if self.verbose:
                print(f'Main loop iteration {iteration}')

            u, dice = self.segmenter.segment(u, self.image, self.plotting)      # Perform segmentation
            new_im = self.reconstructor.reconstruct(self.image, u)      # Perform reconstruction
            self.image.update_image(new_im)     # Update the reconstructed image
        
        end = timer()

        if self.verbose:
            print(f'Runtime: {end - start}')

        plt.ioff()

        return u, self.u0, new_im, dice

if __name__ == '__main__':
    image = Image('heart')
    optimizer = Joint_Optimizer(heart_params_seg, heart_params_recon, image, iterations = 5, verbose = True, plotting = True)
    u, im = optimizer.run()
    plt.imshow(im)
    plt.show()


        
        
