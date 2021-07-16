import numpy as np
import cv2

img = cv2.imread('32.jpg')
cv2.imshow("src", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("thresh", thresh)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("open", opening)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# sure_bg = opening
cv2.imshow("surebg", sure_bg)

dist_transform = cv2.distanceTransform(opening, 1, 5) #浮点型色彩空间,计算像素点到背景的距离
dist_transform = cv2.normalize(dist_transform,0,255,cv2.NORM_MINMAX)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
cv2.imshow("distout", sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg) #保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow("unkown", unknown)

# Marker labelling
ret, markers1 = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers1 + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers3 = cv2.watershed(img, markers)
img[markers3 == -1] = [0, 0, 255]

cv2.imshow("img", img)
cv2.waitKey(0)
