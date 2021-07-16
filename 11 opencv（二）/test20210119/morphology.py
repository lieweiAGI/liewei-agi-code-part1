import cv2 as cv

img = cv.imread("4.jpg", 0)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# dst = cv.dilate(img, kernel)  # 膨胀
# dst = cv.erode(img, kernel) # 腐蚀
# dst = cv.morphologyEx(img, cv.MORPH_OPEN, kernel) #开
# dst = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)  # 闭
# dst = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)  # 梯度
# dst = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel) # 顶帽
dst = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel) # 黑帽
cv.imshow('src', img)
cv.imshow('dst', dst)
cv.waitKey(0)
