from PIL import Image
import numpy as np

img = Image.open("pic.jpg")
img.show()
img_data = np.array(img)
print(img_data.shape)
print(img_data)

img_data = img_data.transpose(1,0,2)
print(img_data.shape)
img = Image.fromarray(img_data,"RGB")
img.show()