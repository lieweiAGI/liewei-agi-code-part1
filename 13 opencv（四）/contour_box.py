import cv2
import numpy as np

img = cv2.imread('5.jpg')

imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imggray, 127, 255, 0)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 边界矩形
x, y, w, h = cv2.boundingRect(contours[0])
# img_contour = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 最小矩形
rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
img_contour = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

# 最小外切圆
(x, y), radius = cv2.minEnclosingCircle(contours[0])
center = (int(x), int(y))
radius = int(radius)
# img_contour = cv2.circle(img, center, radius, (255, 0, 0), 2)

cv2.imshow("img_contour", img_contour)
cv2.waitKey(0)