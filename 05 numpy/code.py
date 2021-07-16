from PIL import Image,ImageDraw,ImageFont
import random

class gen_code:
    def rand_char(self):
        return chr(random.randint(65,91))
    def rand_bg(self):
        return (random.randint(0, 160),
                random.randint(0, 160),
                random.randint(0, 160))
    def rand_fig(self):
        return (random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255))
    def gen_pic(self):
        w,h = 240,100
        img = Image.new(size=(w,h),mode="RGB",color=(255,255,255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font="arial.ttf", size=60)
        for i in range(h):
            for j in range(w):
                draw.point((j,i),self.rand_bg())
        for i in range(4):
            draw.text((10+i*60,20),self.rand_char(),self.rand_fig(),font)
        #draw.rectangle((10,20,70,80),outline='red')
        img.save("code.jpg")

a = gen_code()
a.gen_pic()



