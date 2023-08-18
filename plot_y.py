import matplotlib.pyplot as plt
from scipy.io import loadmat
from Image import Image

name = 'observation.eps'
path = f'/Users/johncao/Documents/Caltech_SURF_2023/Poster/Poster_images/{name}'

im = Image('heart', build_y = True)
fig = plt.figure()
plt.axis('off')
plt.imshow(im.image)
plt.savefig(path, bbox_inches='tight')
plt.show()