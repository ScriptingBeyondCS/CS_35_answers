import numpy as np
from matplotlib import pyplot as plt
import cv2

def show_image(image="share_road.png"):
    
    raw_image = cv2.imread(image,cv2.IMREAD_COLOR) 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.show()

def rewrite(image="share_road.png"):
    raw_image = cv2.imread(image,cv2.IMREAD_COLOR) 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    letter_t = image[190:270, 150:200]
    
    for row in range(310,390):
        for col in range(130,180):
            image[row,col] = letter_t[row-310,col-130]

    plt.imshow(image)
    plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("rewritten_image.png", image)
