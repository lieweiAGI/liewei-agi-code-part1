import cv2

img = cv2.imread('28.jpg', 0)
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

M = cv2.moments(contours[0])  # 矩
cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
print("重心:", cx, cy)
print(M)
area = cv2.contourArea(contours[0])
print("面积:", area)

perimeter = cv2.arcLength(contours[0], True)
print("周长:", perimeter)