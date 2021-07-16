from torch.utils.data import Dataset,DataLoader
import os,numpy,torch
from PIL import Image,ImageDraw
import conv_net

class yellow_dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(os.listdir(path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        #图片路径
        img_path = os.path.join(self.path, self.dataset[item])
        #打开图片
        img_data = Image.open(img_path)
        #获取标签
        label = torch.Tensor(numpy.array(self.dataset[item].split(".")[1:5],dtype=numpy.float32))
        #归一化
        label = label/300
        #通道统一
        img_data = img_data.convert("RGB")
        #归一化，取均值化
        img_data = torch.Tensor(numpy.array(img_data)/255 - 0.5)
        return img_data, label

if __name__ == '__main__':
    # label = "1.105.50.232.177.png"
    # print(torch.Tensor(numpy.array(label.split(".")[1:5],dtype=numpy.float32)))
    net = conv_net.conv_Net()
    net.load_state_dict(torch.load("points/30.t"))
    image = Image.open('data/30.116.184.186.254.png')
    img = numpy.array(image)/255-0.5
    print(img.shape)
    img = img.transpose((2,0,1))
    print(img.shape)
    img = torch.Tensor(img.reshape((1,3,300,300)))
    out = net(img)
    print(out)
    x1,y1,x2,y2 = out[0]*300
    print(x1,y1,x2,y2)



    # mydata = yellow_dataset("data")
    # img, label = mydata[0][0], mydata[0][1]
    # print(img.shape)
    # print(label.shape)
    # print(label)
    # print(img)
    # img = numpy.array((img+0.5)*255,dtype=numpy.uint8)
    # img = Image.fromarray(img, "RGB")
    # # img.show()
    # label = label*300
    # # print(label)
    draw = ImageDraw.Draw(image)
    draw.rectangle((x1,y1,x2,y2),outline="red")
    image.show()
    # dataloarder = DataLoader(mydata, 10, True)

    # for i, (img, tag) in enumerate(dataloarder):

