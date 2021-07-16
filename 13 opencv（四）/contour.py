import cv2

img = cv2.imread('5.jpg')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imggray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(len(contours[0]))
print(len(contours))
print(hierarchy)
img_contour = cv2.drawContours(img, contours,-1, (0, 0, 255), 2) #三通道才能显示轮廓
cv2.imshow("img_contour", img_contour)
cv2.waitKey(0)