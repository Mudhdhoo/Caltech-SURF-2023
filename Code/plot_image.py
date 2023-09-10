import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from Image import Image

plt.rcParams["font.family"] = "Times New Roman"

im = Image('heart', build_y = False)
plt.imshow(im.image)
plt.title('Ground Truth Image', fontweight = 'bold', fontsize = 20)
plt.axis('off')
plt.savefig(f'/Users/johncao/Documents/Caltech_SURF_2023/Poster/Poster_images/clean_im.eps', bbox_inches='tight')
plt.show()
