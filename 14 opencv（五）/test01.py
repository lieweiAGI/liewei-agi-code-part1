import cv2
import matplotlib.pyplot as plt
img = cv2.imread("14.jpg", 0)
cv2.imshow("src",img)
img = cv2.distanceTransform(img, 1, 5)
cv2.imshow("img", img)
print(img)
print(img[150,40:120])
dist_output = cv2.normalize(img,0,255,cv2.NORM_MINMAX)
cv2.imshow("dstouput",dist_output)
ret, img = cv2.threshold(img, 0.8 * img.max(), 255, 0)
cv2.imshow("1",img)
cv2.waitKey()
