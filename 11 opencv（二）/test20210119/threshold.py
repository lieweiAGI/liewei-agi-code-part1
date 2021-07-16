import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('1.jpg',0)
ret,thresh1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh3 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
cv2.imshow("Binary1", thresh1)
cv2.imshow("Binary2", thresh2)
# cv2.imshow("Binary3", thresh3)

cv2.waitKey()
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#
# plt.show()