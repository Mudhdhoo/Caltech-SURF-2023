import numpy as np
from Image import Image
import matplotlib.pyplot as plt

class Segmenter:
    def __init__(self, image: Image) -> None:
        self.image = image   # The image to segment
        self.u = self.init_u0()     

    def init_u0(self):
        """
        Creates the initial segmentation of the image.
        """
        M1, N1 = self.image.image_size[0], self.image.image_size[1] # 2D dimension of image
        circle_center = np.array([M1/2, N1/2])  
        circle_radius = N1/5    # Hardcoded atm, can change to be dynamic later
        phi0 = np.zeros([M1, N1])
        for i in range(0,M1):     # Iterate rows
            for j in range(0,N1):     # Iterate columns
                phi0[i][j] = circle_radius - np.sqrt(np.sum(([i, j]-circle_center)**2))
        u0 = np.heaviside(phi0, 0)
        return u0
    
    def segment(self):
        pass

    def __MBO(self):
        """
        Implements the MBO scheme 
        """
        print('hi')

if __name__ == '__main__':
    im = Image('image')
    seg = Segmenter(im)
    u0  = seg.init_u0()
    # plt.imshow(u0)
    # plt.show()


