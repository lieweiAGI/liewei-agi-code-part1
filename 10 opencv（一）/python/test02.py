from PIL import Image
import numpy

img = Image.open("pic.jpg")
x = numpy.asarray(img)
print(x.shape)
h, w, c = x.shape
x = x.reshape(920, 2, 345, c)
img = Image.fromarray(x[1], "RGB")
img.show()
# x = x.swapaxes(1,2)
x = x.transpose(1,0,2,3)
# print(x.shape)
# x = x.reshape(4,460,345,3)
# for i in range(4):
#     img = Image.fromarray(x[i])
#     img.save("{}.jpg".format(i))
# img1.show()
img = Image.fromarray(x[0], "RGB")
img.show()