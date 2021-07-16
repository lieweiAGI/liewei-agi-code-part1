from PIL import Image,ImageFilter

img = Image.open(r"pic.jpg") #用相对路径打开图片
#img.show() #展示图片，用电脑自带的查看图片方法打开图片

# w,h = img.size #获取尺寸
# print(w)
# print(h)
#
# bands = img.getbands() #获取通道
# print(bands)
#
# mode = img.mode #获取模式
# print(mode)
#
# img1 = img.resize((200,200)) #缩放图片
# print(img1.size)
# img1.show()
# img.show()
#
# img.thumbnail((600,600),resample=Image.ANTIALIAS) #等比缩放图片
# print(img.size)
# img.show()

# img = img.crop((200,200,250,250)) #抠图
# img.show()

# img = img.rotate(-90) #旋转
# img.show()

# img = img.filter(ImageFilter.BLUR) #过滤器
# img = img.filter(ImageFilter.BoxBlur(radius=10))
# img = img.filter(ImageFilter.CONTOUR)
# img = img.filter(ImageFilter.MaxFilter)
# img.show()
# img = img.filter(ImageFilter.DETAIL)
# img = img.filter(ImageFilter.EDGE_ENHANCE)
# img = img.filter(ImageFilter.EMBOSS)
# img.show()

#加Logo
logo = Image.open(r"logo.jpg")
print(logo.mode)
logo = logo.resize((50,50),resample=Image.ANTIALIAS)
img.paste(logo,box=(450,291))
img.show()

img.save("nana.jpg")