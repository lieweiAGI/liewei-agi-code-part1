import cv2
import numpy as np
import matplotlib.pyplot as plt
roi = cv2.imread('10.jpg')
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
target = cv2.imread('9.jpg')
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsvt",hsvt)

#草坪
roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

#归一化
cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)

#取反向投影
dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 20)
# cv2.imshow("dst1", dst)

# 补空洞
disc = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst = cv2.filter2D(dst, -1, disc)
# cv2.imshow("dst2",dst)

#二值化处理
ret, thresh = cv2.threshold(dst, 50, 255, 0)
# cv2.imshow("thresh1,", thresh)

#合并通道变为3通道
thresh = cv2.merge((thresh, thresh, thresh))
# cv2.imshow("thresh2,", thresh)
res = cv2.bitwise_and(target, thresh)

cv2.imshow('img', res)
cv2.waitKey(0)
