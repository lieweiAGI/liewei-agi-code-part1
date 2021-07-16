import cv2
from PIL import Image
import numpy

# img = numpy.random.randint(0,255,(200,300,3))
# cv2.imwrite("image/save.jpg",img)

# img = numpy.zeros((200,300,3),dtype=numpy.uint8)
# # img[:,:,0] = 255
# img[...,0] = 255
# print(img.shape)
# a = Image.fromarray(img)
# a.save("image/PIL_save.jpg")
# cv2.imwrite("image/save.jpg",img)

img = cv2.imread("pic.jpg")
# img = Image.fromarray(img)
# img.show()
img = img[...,::-1] #反序
img = Image.fromarray(img)
img.show()

