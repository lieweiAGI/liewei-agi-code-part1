from PIL import Image,ImageDraw

img = Image.open("img/000002.jpg")
draw = ImageDraw.Draw(img)
#draw.rectangle((95,71,95+ 226,71+ 313),outline="red",width=2)
draw.rectangle((72,  94, 72+221,94+ 306),outline="red",width=2)
img.show()