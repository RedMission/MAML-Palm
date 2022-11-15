import torch
from torchsummary import summary
from learner import Learner # 自定义类
from torchvision.models import vgg16  # 以 vgg16 为例
from    torch import nn

from learner_inception_new import Learner_inception_new

n_way=5
config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),

        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),

        ('flatten', []),
        ('linear', [n_way, 32 * 5 * 5])
    ]
config_inception = [  # 参数按照网上的例子来的 格式：[('类型',[参数，参数，…]),()]
        ('conv2d', [10, 3, 5, 5, 1, 0]),
        ('max_pool2d', [2, 2, 0]),
        ('relu', [True]),

        # inception 输入为10，与conv1 中的10对应
        ('branch1x1', [16, 10, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 10, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]),

        ('branch3x3_1', [16, 10, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 10, 1, 1, 1, 0]),

        # inception1结束
        # ————————下一个卷积层————————
        ('conv2d', [88, 10, 5, 5, 1, 0]),
        ('max_pool2d', [2, 2, 0]),
        ('relu', [True]),

        # inception2 输入为20，与conv2 中输出的20对应
        ('branch1x1', [16, 20, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 20, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]),

        ('branch3x3_1', [16, 20, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 20, 1, 1, 1, 0]),
        # __________inception2结束__________
        ('flatten', []),
        ('linear', [10, 1408])
    ]



#
# net = Learner(config)
# # print(net)
# print(net.parameters())
# print("_______________net2__________________")
# net2 = Learner_inception()
# # print(net2)
# print(net2.parameters())

print("_______________net3__________________")
net3 = Learner_inception_new(config_inception)
print(net3)
# print(net3.parameters())
