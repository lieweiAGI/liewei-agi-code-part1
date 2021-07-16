import cv2
import numpy as np
img = cv2.imread("11.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([1, 180, 110])
upper_blue = np.array([200, 255, 200])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
print(mask.shape)
# res = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('frame', img)
cv2.imshow('mask', mask)
cv2.waitKey(0)