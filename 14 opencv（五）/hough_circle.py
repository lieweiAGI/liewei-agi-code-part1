# image, 输入一个图像
# method, 检测圆的方法，只有实现了霍夫梯度
# dp, 分辨率
# minDist, 圆中心的最小距离
# param1=None, 用于做边缘检测的梯度值
# param2=None, 投票值
# minRadius=None, 最小半径
# maxRadius=None, 最大半径

import cv2
import numpy as np

image = cv2.imread("30.jpg")
dst = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
circle = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT,1,10, param1=40, param2=20, minRadius=20, maxRadius=25)
print(circle)
if not circle is None:
    circle = np.uint16(np.around(circle))
    for i in circle[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 1)

cv2.imshow("circle", image)
cv2.waitKey(0)
cv2.morphologyEx()