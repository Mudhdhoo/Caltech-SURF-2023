import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from PIL import Image as Im

class Image:
    """
    Class for representing and manipulating the image.
    """
    def __init__(self, image, scale = 1, blur = 0, noise_std = 0.15, blur_std = 0.4, is_grey = 0, is_binary = 0, KER = 1, build_y = True, ground_truth = None) -> None:
        # Image set-up
        if type(image) == str:
            image = loadmat(os.path.join('images',image))[image]
            self.image = image / np.max(image)       # Normalize image to [0,1]
        else:
            self.image = image / np.max(image)
            
        self.image_size = self.image.shape
        self.color = False
        if len(self.image_size) > 2:
            self.color = True
        self.blur = blur
        self.noise_std = noise_std
        self.blur_std = blur_std
        self.is_grey = is_grey
        self.is_binary = is_binary
        if build_y:
            self.image = self.build_y()
        self.make_feature_map()
        self.y = self.image
        self.ground_truth = ground_truth

        # Feature map parameters
        self.KER = KER

    def blur_image(self):
        # TODO for reconstruction
        pass

    def noise_image(self, image):
        """
        Add gaussian noise to the image.
        """
        if self.color:
            M, N, L = image.shape
            noise = np.random.normal(0, self.noise_std, [M,N,L])
        else:
            M, N = image.shape
            noise = np.random.normal(0, self.noise_std, [M,N])
        
        return image + noise

    def build_y(self):
        """
        Create noisy/distorted observation y = T(x) + e.
        """
        y = self.image

        # Blur the image
        if self.blur:
            y = self.blur_image()

        # Add noise
        y = self.noise_image(y)

        return y
        
    def resize_image(self, image, scale):
        """
        Resizes the image.
        """
        im = Im.fromarray(image)
        M = round(image.shape[1]*scale)
        N = round(image.shape[0]*scale)
        resized_im = im.resize([M, N])
        resized_im = np.array(resized_im)

        return resized_im

    def make_feature_map(self):
        """
        Identity feature map. Flattens out the 2D image into an M*N x 1 vector of pixel intensities.
        """
        if self.color:
            M, N, L = self.image.shape
            self.J = self.image.reshape(M*N, L)
        else:
            M, N = self.image.shape
            self.J = self.image.reshape(M*N, 1)

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
    im = loadmat('u0.mat')['u0']
    im = (im*255).astype('uint8')
    print(im.dtype)
    #im = Image(im, ground_truth = None, build_y = False, scale=1)
    plt.imshow(im)
    plt.show()
    #im.show()
    