from PIL import Image
import numpy as np

img = Image.open("pic.jpg")
# img.show()
print(type(img))
print(img)
x = np.asarray(img) #图片转化为数组
print(x)
print(x.shape)
img = Image.fromarray(x) #数组转换为图片
img.show()
