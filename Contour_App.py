import tkinter as tk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
matplotlib.use("TKAgg")

class Contour_App:
    """
    GUI for creating a hand-drawn initial segmentation of the image.
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Initial Segmentation")

        self.canvas = tk.Canvas(self.root, bg = "white")
        self.canvas.pack(fill = tk.BOTH, expand = True)

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.drawing = False
        self.points = []  # Store the points of the current contour

        self.clear_button = tk.Button(self.root, text = "Clear Canvas", command = self.clear_canvas)
        self.clear_button.pack()

        self.load_image_button = tk.Button(self.root, text = "Load Image", command = self.load_image)
        self.load_image_button.pack()

        self.return_button = tk.Button(self.root, text = "Begin Segmentation", command = self.get_segmentation)
        self.return_button.pack()

        self.image = None
        self.image_id = None

    def run(self):
        self.root.mainloop()

        return self.u0, self.image_array

    def start_drawing(self, event):
        self.drawing = True
        self.points = []  # Start a new contour
        x, y = event.x, event.y
        self.points.append((x, y))

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.points.append((x, y))
            self.canvas.delete("current_contour")  # Clear the previous contour
            self.canvas.create_polygon(self.points, fill = '', outline = "magenta", tags = "current_contour", dash=(5,5))

    def stop_drawing(self, event):
        if self.drawing:
            self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        if self.image_id:
            self.canvas.delete(self.image_id)
            self.image_id = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.ppm *.pgm *.pbm *.tiff *.tif")])
        if file_path:
            self.clear_canvas()
            self.image = Image.open(file_path)
            self.image_array = np.array(self.image)
            self.image = ImageTk.PhotoImage(self.image)
            self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.canvas.config(width=self.image.width(), height=self.image.height())

    def get_segmentation(self):
        if self.image:
            width = self.image.width()
            height = self.image.height()
        else:
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()

        segmentation = np.zeros((height, width, 3), dtype=np.uint8) + 255

        # Iterate through all canvas items and draw them on the image
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 4:
                # Convert coordinates to integers
                coords = [int(coord) for coord in coords]
                # Fill the contour with black
                cv2.fillPoly(segmentation, [np.array(coords).reshape((-1, 2))], (0, 0, 0))

        self.u0 = segmentation[:,:,0] < 1
        self.u0 = self.u0.astype('int')
        self.root.destroy()

if __name__ == "__main__":
    app = Contour_App()
    init_seg, image = app.run()
    plt.imshow(init_seg*image)
    plt.show()
