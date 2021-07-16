#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2

# 读取图片
rawImage = cv2.imread("chepai.jpg")

# 高斯模糊，将图片平滑化，去掉干扰的噪声
image = cv2.GaussianBlur(rawImage, (3, 3), 1)

# 图片灰度化
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Sobel算子（X方向）
Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)

# Sobel_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(Sobel_x)  # 转回uint8

# cv2.imshow("img",rawImage)

# absY = cv2.convertScaleAbs(Sobel_y)
# dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

image = absX
# 二值化：图像的二值化，就是将图像上的像素点的灰度值设置为0或255,图像呈现出明显的只有黑和白
ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

# 闭操作：闭操作可以将目标区域连成一个整体，便于后续轮廓的提取。
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)
cv2.imshow("win1",image)
# 膨胀腐蚀(形态学处理)
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
image = cv2.dilate(image, kernelX)
image = cv2.erode(image, kernelX)
image = cv2.erode(image, kernelY)
image = cv2.dilate(image, kernelY)
cv2.imshow("win2",image)
# cv2.imshow("xxx",image);
# cv2.waitKey(0);
# exit()

# 平滑处理，中值滤波
image = cv2.medianBlur(image, 15)
cv2.imshow("...2.", image)

# 查找轮廓
contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for item in contours:
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    if weight > (height * 2) and height > 5:
        # 裁剪区域图片
        chepai = rawImage[y:y + height, x:x + weight]
        cv2.imshow('chepai' + str(x), chepai)

# 绘制轮廓
image = cv2.drawContours(rawImage, contours, -1, (0, 0, 255), 3)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
