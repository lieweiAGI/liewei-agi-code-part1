import cv2
import numpy as np

img = cv2.imread("25.jpg")
cv2.imshow("img", img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img1 = cv2.blur(img,(3,3))
# img1 = cv2.GaussianBlur(img,(5,5),0)
# cv2.imshow("Gblur",img1)
# img1 = cv2.GaussianBlur(img,(5,5),5)
# cv2.imshow("2",img1)
# img1 = cv2.GaussianBlur(img1,(5,5),5)
# cv2.imshow("blur",img1)
# dst = cv2.medianBlur(img, 9)
# cv2.imshow("mid", dst)
# img = cv2.bilateralFilter(img, 13 , 75, 75)
# cv2.imshow("bilateral", img)
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)
cv2.imshow("sobelx", sobelx)
cv2.imshow("sobely", sobely)
sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5,1)
cv2.imshow("x+y", sobel)
# kernel = np.array([[0, -1, 0], [-1, 7, -1], [0, -1, 0]], np.float32) #定义一个核
# dst = cv2.filter2D(img, -1, kernel=kernel)

# lap = cv2.Laplacian(img, -1, ksize=3)
# cv2.imshow("laplacian", lap)
# scharrx = cv2.Scharr(img,-1,1,0)
# cv2.imshow("scharrx", scharrx)
# kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9],
#                    [1/9, 1/9, 1/9]],np.float32)
# img = cv2.filter2D(img,-1,kernel)

# dst = cv2.GaussianBlur(img, (5, 5), 0)
# dst = cv2.addWeighted(img, 2, dst, -1, 0)
# cv2.imshow("USM", dst)

# cv2.imshow("filter", img)
# kernel = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], np.float32) # 定义一个核
# dst = cv2.filter2D(img, -1, kernel=kernel)
# cv2.imshow("1",img1)
cv2.waitKey()
