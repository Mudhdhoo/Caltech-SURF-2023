import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

class Image:
    """
    Class for representing and manipulating the image.
    """
    def __init__(self, image, scale = 1, blur = 0, noise_std = 0.15, blur_std = 0.4, is_grey = 0, is_binary = 0, KER = 1, build_y = True, ground_truth = None) -> None:
        # Image set-up
        if type(image) == str:
            image = loadmat(os.path.join('images',image))[image]
            image = image / np.max(image)       # Normalize image to [0,1]

        self.image = image
        self.image_size = self.image.shape
        self.blur = blur
        self.noise_std = noise_std
        self.blur_std = blur_std
        self.is_grey = is_grey
        self.is_binary = is_binary
        if build_y:
            self.image = self.build_y()
        #self.J = self.make_feature_map()
        self.make_feature_map()
        self.y = self.image
        self.ground_truth = ground_truth

        # Feature map parameters
        self.KER = KER

    def blur_image(self):
        # TODO for reconstruction
        pass

    def noise_image(self, image):
        M, N = image.shape
        noise = np.random.normal(0, self.noise_std, [M,N])
        
        return image + noise

    def build_y(self):
        y = self.image

        # Blur the image
        if self.blur:
            y = self.blur_image()

        # Add noise
        y = self.noise_image(y)

        return y
        
    def resize_image(self):
        # TODO
        pass

    def make_feature_map(self):
        """
        Identity feature map. Flattens out the 2D image into an M*N x 1 vector of pixel intensities.
        """
        M = self.image.shape[0]
        N = self.image.shape[1]
        self.J = self.image.reshape(M*N, 1)

        #return 

    def update_image(self, new_image):
        """
        Updates the current image with a new one and creates a new feature map.
        """
        self.image = new_image
        self.make_feature_map()

    def show(self):
        """
        Display the image.
        """
        plt.imshow(self.image)
        plt.show()

if __name__ == '__main__':
    gt = loadmat(os.path.join('images','heart_truth'))['groundtruth']
    im = Image('heart', ground_truth = gt)
    plt.imshow(gt)
    plt.show()
    #im.show()
    