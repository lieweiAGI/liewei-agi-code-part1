from PIL import Image,ImageDraw

img = Image.open("data/12.48.12.184.148.png")
draw = ImageDraw.Draw(img)
draw.rectangle((48,12,184,148),outline="red")
img.show()