import cv2

img = cv2.imread(r"12.jpg")
print(img.size)
img_down = cv2.pyrDown(img)
print(img_down.size)
img_up = cv2.pyrUp(img_down)
print(img_up.size)
img_new = cv2.subtract(img, img_up)
#为了更容易看清楚，做了个提高对比度的操作
img_new = cv2.convertScaleAbs(img_new, alpha=20, beta=0)
cv2.imshow("img_LP", img_new)
cv2.waitKey(0)