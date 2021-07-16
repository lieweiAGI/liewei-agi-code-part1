import cv2
import numpy as np

src = cv2.imread('1.jpg')
rows, cols, channel = src.shape
# M = np.float32([[1, 0, 50], [0, 1, 50]]) #平移变换
# M = np.float32([[0.5, 0, 0], [0, 0.5, 0]]) #缩放
# M = np.float32([[-0.5, 0, cols // 2], [0, 0.5, 0]]) #复合变换
# M = np.float32([[1, 0.5, 0], [0, 1, 0]]) #斜拉
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 0.7)
dst = cv2.warpAffine(src, M, (cols, rows))
cv2.imshow('src pic', src)
cv2.imshow('dst pic', dst)
cv2.waitKey(0)