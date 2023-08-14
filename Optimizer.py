import matplotlib.pyplot as plt
import numpy as np
from Segmenter import Segmenter
from Reconstructor import Reconstructor
from Image import Image
from Parameters import *
from params import *

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

    def __init__(self, seg_params:Segmentation_Params, recon_params:Reconstruction_Params, image:Image, iterations:int, verbose = False) -> None:
        self.segmenter = Segmenter(seg_params)
        self.reconstructor = Reconstructor(recon_params, 'TV')
        self.image = image
        self.iterations = iterations
        self.verbose = verbose

    def run(self):
        """
        compute cheap reconstruction
        compute initial contour
        for iteration in iterations do:
            compute segmentation
            compute reconstruction based on segmentaion
            update image with new reconstruction
        """
        # Pre-processing cheap reconstruction
        y = self.image.image
        Im0 = self.reconstructor.cheap_reconstruction(y)
        self.image.update_image(Im0)

        # Initial contour & segmentation
        u0 = self.segmenter.init_u0(self.image)

        # Main optimization loop
        u = u0
        for iteration in range(0, self.iterations):
            if self.verbose:
                print(f'Main loop iteration {iteration}')

            u = self.segmenter.segment(u, self.image)      # Perform segmentation
            new_im = self.reconstructor.reconstruct(self.image, u)      # Perform reconstruction
            self.image.update_image(new_im)     # Update the reconstructed image
        
        plt.ioff()
    
        return u, new_im

if __name__ == '__main__':
    image = Image('heart')
    optimizer = Joint_Optimizer(heart_params_seg, recon_params, image, iterations = 5, verbose = True)
    u, im = optimizer.run()
    plt.imshow(u)
    plt.show()

        
        
