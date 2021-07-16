import cv2

img = cv2.imread('26.jpg')

imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imggray,127,255,0)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = cv2.convexHull(contours[0])
print(cv2.isContourConvex(contours[0]), cv2.isContourConvex(hull))
#False True
#说明轮廓曲线是非凸的，凸包曲线是凸的

img_contour= cv2.drawContours(img, [hull], -1, (0, 0, 255), 3)

cv2.imshow("img_contour", img_contour)
cv2.waitKey(0)