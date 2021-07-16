import cv2

img = cv2.imread("31.jpg", 0)
img = cv2.bilateralFilter(img,8,50,50)
img1 = cv2.Canny(img, 0, 200)
img2 = cv2.Canny(img, 30, 150)
cv2.imshow("canny1",img1)
cv2.imshow("canny", img2)
cv2.waitKey()