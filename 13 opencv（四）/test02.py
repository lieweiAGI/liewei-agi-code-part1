import cv2
import numpy as np
img = cv2.imread('14.jpg')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imggray, 127, 255, 0)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(imggray.shape, np.uint8)
img = cv2.drawContours(img, [contours[0]], -1, (0,0,255), 2)
cv2.imshow("img", img)
cv2.drawContours(mask, [contours[0]], -1, 255, -1)

print(mask)
cv2.imshow("mask", mask)
cv2.waitKey(0)