import numpy as np
from PIL import Image

img = Image.open("pic.jpg")
img_data = np.array(img)
print(img_data)