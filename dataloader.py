import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
import torchvision.transforms as transforms


class Dataset(Dataset):

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.transform(self.x[idx]),self.y[idx]

def modeldataloader(raw_data, num_of_classes, shuffle, batch_size):
    mid_pixel_value = 1.0 / 2
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3), # 将单通道图像转换为三通道灰度图像
            transforms.Resize(84),  # 图像尺寸太大会内存溢出
            transforms.ToTensor(),
            transforms.Normalize(
                (mid_pixel_value,) * 3, (mid_pixel_value,) * 3
            ),
        ]
    )
    x = []
    y = []
    for i in range(num_of_classes):
        x_data = list(raw_data[i])
        np.random.shuffle(x_data) # 某类选择某张图作为x
        # 随机第一个作为模板
        x.append(x_data[0]) # (128, 128, 1) ndarray
        y.append(i)  #
    # 实例化对象
    train_dataset = Dataset(x, y, transform)
    # print("dataset长度：",train_dataset.__len__())
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
def normaldataloader(raw_data, num_of_classes, shuffle, batch_size):
    mid_pixel_value = 1.0 / 2
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3), # 将单通道图像转换为三通道灰度图像
            transforms.Resize(84),  # 图像尺寸太大会内存溢出
            transforms.ToTensor(),
            transforms.Normalize(
                (mid_pixel_value,) * 3, (mid_pixel_value,) * 3
            ),
        ]
    )
    x = []
    y = []
    for i in range(num_of_classes):
        for x_data in list(raw_data[i]):
            x.append(x_data) # (128, 128, 1) ndarray
            y.append(i)  #
    # 实例化对象
    train_dataset = Dataset(x, y, transform)
    print("dataset长度：",train_dataset.__len__())
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

if __name__ == '__main__':
    raw_data = np.load("F:\jupyter_notebook\DAGAN\datasets\IITDdata_left_PSA_2+MC+SC+W_6.npy", allow_pickle=True).copy() #numpy.ndarray

    num_of_classes = 10
    batch_size = 1
    # 创建训练数据加载器
    dataloader = normaldataloader(raw_data, num_of_classes,True, batch_size)

    for i,item  in enumerate(dataloader):
        data, label = item
        print('data:', data.shape)
        print('label:', label)
