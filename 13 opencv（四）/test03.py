import cv2
img = cv2.imread('1.jpg')
cv2.imshow("img", img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.imshow("hist",hist)
cv2.waitKey()