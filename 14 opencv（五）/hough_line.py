# image  输入一个检测过边缘的二值化图像(单通道)
# rho  对应精度为1，可以想象成坐标系精度
# theta  对应精度为1°，可以想象成坐标系精度
# threshold  投票值，累加器

import cv2
import numpy as np

image = cv2.imread("32.jpg")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(image_gray, 100, 200)
cv2.imshow("edges",edges)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
print(lines.shape)
#y=ax+b --> b=-ax+y
#rho = xcos + ysin
#垂点(rho*cos,rho*sin)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
    y1 = int(y0 + 1000 * (a))  # 直线起点纵坐标
    x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
    y2 = int(y0 - 1000 * (a))  # 直线终点纵坐标
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow("image_lines", image)

cv2.waitKey(0)
