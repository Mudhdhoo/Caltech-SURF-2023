import numpy as np
import os
from scipy.io import loadmat
from PIL import Image

# im = loadmat(os.path.join('images', 'heart'))['heart']#*255
# im = Image.fromarray(im)
# im.save('./images/heart.png')

# im = loadmat(os.path.join('images', 'cow'))['cow']*255
# im = Image.fromarray(im.astype('uint8'), 'RGB')
# im.save('./images/cow.png')

# im = loadmat(os.path.join('images', 'heart_truth.mat'))['groundtruth']*255
# im = Image.fromarray(im)
# im.save('./images/heart_truth.png')

im = loadmat(os.path.join('images', 'cow_truth.mat'))['groundtruth']*255
im = Image.fromarray(im)
im.save('./images/cow_truth.png')