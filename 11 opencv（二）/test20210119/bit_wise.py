import cv2
img1 = cv2.imread('1.jpg') #美女图
img2 = cv2.imread('6.jpg') #logo

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols] #美女图上对应logo图的位置
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow("mask", mask)
img1_bg = cv2.bitwise_and(roi, roi, mask=mask) #美女图上扣字
cv2.imshow("img1_mask", img1_bg)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv) #logo图扣字
cv2.imshow("img2_fg",img2_fg)
dst = cv2.add(img1_bg, img2_fg)
cv2.imshow('res',dst)
cv2.waitKey(0)