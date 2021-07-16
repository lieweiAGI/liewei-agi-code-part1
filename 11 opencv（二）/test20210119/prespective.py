import cv2
import numpy as np

img = cv2.imread("2.jpg")

pts1 = np.float32([[25, 30], [179, 25], [12, 188], [189, 190]])
pts2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])

M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (200, 200))

cv2.imshow("src", img)
cv2.imshow("dst", dst)

cv2.waitKey(0)