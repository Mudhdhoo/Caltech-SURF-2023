import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from Image import Image

plt.rcParams["font.family"] = "Times New Roman"
gt = loadmat(os.path.join('images','heart_truth'))['groundtruth']
im = Image('heart', build_y = False, ground_truth = gt)
plt.imshow(im.ground_truth)
plt.axis('off')
plt.title('Ground Truth Segmentation', fontweight = 'bold', fontsize = 20)
plt.savefig(f'/Users/johncao/Documents/Caltech_SURF_2023/Poster/Poster_images/truth.eps', bbox_inches='tight')
plt.show()
