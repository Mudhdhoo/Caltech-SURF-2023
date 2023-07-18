import numpy as np
import time
import matplotlib.pyplot as plt

#plt.ion()
fig = plt.figure()

for i in range(0,10):
    im = np.random.rand(256,256)
    plt.imshow(im)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)
plt.show()