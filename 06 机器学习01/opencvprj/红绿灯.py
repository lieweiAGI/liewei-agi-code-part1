import cv2
import numpy as np

#红绿灯检测，传入图片返回侦测出的矩形框和轮廓
def tl_detection(rawImage):
    # 高斯模糊，将图片平滑化，去掉干扰的噪声
    image = cv2.GaussianBlur(rawImage, (5, 5), 3)

    # 图片灰度化
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobel算子（X方向）
    Sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    # Sobel_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(Sobel_x)  # 转回uint8
    ret, image = cv2.threshold(absX, 127, 255, cv2.THRESH_OTSU)


    # 闭操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX)

    # 膨胀腐蚀
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)
    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)

    # 中值滤波
    image = cv2.medianBlur(image, 17)

    # 轮廓提取
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # red色调
    lower_hsv_red = np.array([139, 56, 100])
    upper_hsv_red = np.array([179, 255, 255])
    # green色调
    lower_hsv_green = np.array([32, 73, 169])
    upper_hsv_green = np.array([92, 255, 255])

    # 侦测框列表
    rectlist = []

    # 提取边界矩形，从中判断合适的建议框
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        # 筛选边界矩形
        if height < (weight * 2.5) and height > weight * 2 and weight > 10:
            # 裁剪区域图片
            a = rawImage[y:y + height, x:x + weight]
            cv2.imshow('traffic light' + str(x), a)
            a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
            mask_red = cv2.inRange(a, lowerb=lower_hsv_red, upperb=upper_hsv_red)
            mask_green = cv2.inRange(a, lowerb=lower_hsv_green, upperb=upper_hsv_green)

            # 根据掩码判断颜色，红灯用红框，绿灯用绿框，没判断出颜色用蓝框
            if np.max(mask_red) == 255:
                rectlist.append((x, y, x + weight, y + height, (0, 0, 255)))
            elif np.max(mask_green) == 255:
                rectlist.append((x, y, x + weight, y + height, (0, 255, 0)))
            else:
                rectlist.append((x, y, x + weight, y + height, (255, 0, 0)))
    return rectlist, contours

if __name__ == '__main__':
    image = cv2.imread("3.jpg")
    rectlist, contours = tl_detection(image)
    for i in rectlist:
        cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), i[4], thickness=2)
    # image = cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
    cv2.imshow('image', image)
    cv2.waitKey()
