# from tkinter import *
# import os
# import numpy as np
# from scipy.io import loadmat
# from PIL import Image, ImageTk

# def get_x_y(event):
#     global prev_x, prev_y
#     prev_x, prev_y = event.x, event.y

# def draw(event):
#     global prev_x, prev_y
#     canvas.create_line((prev_x, prev_y, event.x, event.y), fill = 'red', width=5)
#     prev_x, prev_y = event.x, event.y

# image = loadmat(os.path.join('images','heart'))['heart']    # Grey scale value
# app = Tk()
# app.geometry(f"{image.shape[0]}x{image.shape[1]}")
# canvas = Canvas(app)
# canvas.pack(anchor='nw', fill='both', expand=1)

# canvas.bind('<Button-1>', get_x_y)
# canvas.bind('<B1-Motion>', draw)

# image = ImageTk.PhotoImage(image=Image.fromarray(image))
# canvas.create_image(0,0, image = image, anchor = 'nw')

# app.mainloop()
import cv2
import numpy as np
drawing = False # true if mouse is pressed
ix,iy = -1,-1

# define mouse callback function to draw circle
def draw_curve(event, x, y, flags, param):
   global ix, iy, drawing, img
   if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
   elif event == cv2.EVENT_MOUSEMOVE:
      if drawing == True:
         cv2.circle(img, (x, y), 3,(0, 0, 255),-1)
      elif event == cv2.EVENT_LBUTTONUP:
         drawing = False
         cv2.circle(img, (x, y), 3,(0, 0, 255),-1)

# Create a black image
img = np.zeros((512,700,3), np.uint8)

# Create a window and bind the function to window
cv2.namedWindow("Curve Window")

# Connect the mouse button to our callback function
cv2.setMouseCallback("Curve Window", draw_curve)

# display the window
while True:
   cv2.imshow("Curve Window", img)
   if cv2.waitKey(10) == 27:
      break
cv2.destroyAllWindows()