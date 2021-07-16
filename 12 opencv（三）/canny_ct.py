import cv2

img = cv2.imread("31.jpg", 0)
col, row = img.shape
print(img.shape)
# img = cv2.Laplacian(img,-1)
cv2.imshow('src', img)
img = cv2.convertScaleAbs(img, alpha=9, beta=0) #高亮处理
cv2.imshow('Abs', img)
img = cv2.GaussianBlur(img, (7, 7), 1) #模糊，去噪
cv2.imshow("img",img)
canny = cv2.Canny(img, 150, 200)
# canny = cv2.resize(canny, dsize=(row, col))
cv2.imshow('Canny', canny)
cv2.waitKey(0)