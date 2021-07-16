import cv2
import numpy
from PIL import Image

#读取图片
img = cv2.imread("pic.jpg")
cv2.imshow("picture", img)
cv2.waitKey() #延迟
cv2.destroyAllWindows() #销毁所有窗口


