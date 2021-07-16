import cv2
import matplotlib.pyplot as plt

img = cv2.imread('1.jpg')
# img[...,0]=0
# img[...,1]=0
cv2.imshow("src", img)

img_B = cv2.calcHist([img], [0], None, [256], [0,255])
plt.plot(img_B, label='B', color='b')
# plt.show()
plt.xlim([0,10])
plt.ylim([0,100])
# img_G = cv2.calcHist([img], [1], None, [256], [0, 255])
# plt.plot(img_G, label='G', color='g')

# img_R = cv2.calcHist([img], [2], None, [256], [0, 255])
# plt.plot(img_R, label='R', color='r')

plt.show()
