from tkinter import *
import os
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageTk

image = loadmat(os.path.join('images','heart'))['heart']    # Grey scale value
app = Tk()
app.geometry("256x256")
canvas = Canvas(app)
canvas.pack(anchor='nw', fill='both', expand=1)

image = ImageTk.PhotoImage(image=Image.fromarray(image))
canvas.create_image(0,0, image = image, anchor = 'nw')
app.mainloop()