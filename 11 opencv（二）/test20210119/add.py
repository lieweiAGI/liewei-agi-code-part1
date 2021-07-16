import cv2
import numpy as np

x = np.uint8([250])
y = np.uint8([100])

img1 = cv2.imread("1.jpg")
img2 = cv2.imread("6.jpg")
#
# cv2.add(img1,img2,img2)
# cv2.imshow("add ", img2)
# cv2.waitKey()
dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 30)
dst1 = cv2.addWeighted(img1, 0.9, img2, 0.1, 0)
dst2 = cv2.addWeighted(img1, 0.7, img2, 0.3, 100)
cv2.imshow('dst', dst)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()