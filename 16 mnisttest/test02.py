import torch
import net,cv2

net = net.Net()
net.load_state_dict(torch.load("param/10.t")) #只加载模型参数
#net1 = torch.load("param/10.t") #加载整个模型
net.eval()
#测试
img_data = cv2.imread("MNIST_IMG/TEST/0/0.jpg", 0)
img_data = img_data.reshape(1,-1)
img_data = img_data/255
img_data = torch.Tensor(img_data)

y = net(img_data)
h = torch.argmax(y)
print(y)
print(h)

