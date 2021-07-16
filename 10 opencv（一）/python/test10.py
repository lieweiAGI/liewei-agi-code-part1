import cv2
from PIL import Image
img = cv2.imread("pic.jpg") #BGR
#img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #BGR-->RGB
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# img1 = Image.fromarray(img1)
# img1.show()
cv2.imshow("pic", img1)
cv2.waitKey()