import cv2
import matplotlib.pyplot as plt
img = cv2.imread('19.jpg', 0)
cv2.imshow("src", img)

# dst1 = cv2.equalizeHist(img)
# cv2.imshow("dst1", dst1)

clahe = cv2.createCLAHE(tileGridSize=(8, 8))
dst2 = clahe.apply(img)
cv2.imshow("dst2", dst2)

cv2.waitKey(0)