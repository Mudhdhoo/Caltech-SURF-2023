import os
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Image:
    def __init__(self, image_name, scale = 1, blur = 0, noise_std = 0.01, blur_std = 0.4, is_grey = 0, is_binary = 0) -> None:
        self.image = loadmat(os.path.join('images',image_name))[image_name]
        self.image_size = self.image.shape
        self.scale = scale
        self.blur = blur
        self.noise_std = noise_std
        self.blur_std = blur_std
        self.is_grey = is_grey
        self.is_binary = is_binary

    def blur_image(self):
        # TODO for reconstruction
        pass

    def noise_image(self):
        # TODO for reconstruction
        pass
        
    def resize_image(self):
        pass

    def show(self):
        plt.imshow(self.image)
        plt.show()

if __name__ == '__main__':
    im = Image('image')
    print(im.image_size)