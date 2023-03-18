import os
import cv2
import numpy as np
outputpath = "./test2/task2/"
kernel = np.ones((3,3), np.uint8)

for i in os.listdir("./test/task2/"):
    img = cv2.imread("./test/task2/" + i)
    img = 255 - img
    mask = np.zeros_like(img)
    ret, th = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    erosion = cv2.erode(th, kernel, iterations = 1)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
    cv2.imwrite(f"{outputpath}/{i}", dilation)
cv2.destroyAllWindows()