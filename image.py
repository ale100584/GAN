from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg 
import numpy as np
import cv2

def char2image(c, image_size=(28,28)):
    '''
    Returns a greyscale image showing the input character c
    '''
    
    if not isinstance(c, str) or len(c)>1:
            raise ValueError("c is not a char")

    fig = plt.figure(figsize=(1,1))
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()

    ax.text(0.5,0.5,c, size=60, horizontalalignment='center', verticalalignment='center')
    ax.axis('off')
    plt.close(fig)

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    X = np.fromstring(s, np.uint8).reshape((height, width, 4))
    gray_image = cv2.bitwise_not(cv2.cvtColor(X, cv2.COLOR_BGR2GRAY))
    resized = cv2.resize(gray_image, image_size)/255.

    return resized
