import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
import torch
import os
from PIL import Image


class MyData(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img = Image.open(img_path)  # 读取该图片
        if self.transform:
            img = self.transform(img)

        return img

img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5,), (0.5,))
    ])

root_dir = '../train_img/'
my_dataset = MyData(root=root_dir, transform=img_transform)


def data_load(batch_size):
    return torch.utils.data.DataLoader(
        dataset=my_dataset ,
        batch_size = batch_size,
        shuffle=True)



data_loader = data_load(256)







