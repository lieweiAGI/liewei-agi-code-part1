import matplotlib.pyplot as plt
from PIL import Image

# img = Image.open("pic.jpg")
# plt.imshow(img)
# plt.show()

ax = []
ay = []
plt.ion() #开启实时画图
for i in range(100):
    ax.append(i)
    ay.append(i**2)
    plt.clf() #清空画板上的内容
    # plt.cla() #清空所有内容
    # plt.scatter(ax,ay,c="r",marker=".") #画点图
    plt.plot(ax, ay) #画折线图
    plt.pause(1)
plt.ioff()
