import time

import numpy as np
import torch
from  torch import nn
from torch.utils.tensorboard import SummaryWriter

from learner_inception_new import Learner_inception_new
from dataloader import modeldataloader, normaldataloader
from    torch.nn import functional as F

'''
微调 maml输出部分（但意义不大
'''

def cossimiliarity(a,b):
    # 计算余弦相似度
    dot_product = np.dot(a, b)
    norm_sample = np.linalg.norm(a)
    norm_template = np.linalg.norm(b)
    similarity = dot_product / (norm_sample * norm_template)
    return similarity

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # 创建一个与保存的模型结构相似的实例【没有输出类别的层】
    net_config = [
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
        # ('linear', [10, 32 * 5 * 5])
    ]
    config_inception_Residual_se = [  # 格式：[('类型',[参数，参数，…]),()]
        ('conv2d', [10, 3, 3, 3, 1, 0]),
        ('bn', [10]),
        ('max_pool2d', [2, 2, 0]),
        ('relu', [True]),
        # ————————注意力机制————————
        ('SqueezeExcite', [10]),

        # inception 输入为10，与conv1 中的10对应
        ('branch1x1', [16, 10, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 10, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]),  # 核考虑换成3*3

        ('branch3x3_1', [16, 10, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 10, 1, 1, 1, 0]),
        # 残差项
        ('downsample', [88, 10, 1, 1, 1, 0]),  # 输入不确定；含一个conv2d 一个bn

        ('relu', [True]),

        # inception1结束
        # ————————下一个卷积层————————
        ('conv2d', [20, 88, 3, 3, 1, 0]),  #
        ('bn', [20]),
        ('max_pool2d', [2, 2, 0]),
        ('relu', [True]),

        ('SqueezeExcite', [20]),

        # inception2 输入为20，与conv2 中输出的20对应
        ('branch1x1', [16, 20, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 20, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]),

        ('branch3x3_1', [16, 20, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 20, 1, 1, 1, 0]),
        # 残差项
        ('downsample', [88, 20, 1, 1, 1, 0]),  # 输入不确定；含一个conv2d 一个bn
        ('relu', [True]),
        # __________inception2结束__________
        # ————————下一个卷积层————————
        ('conv2d', [30, 88, 3, 3, 1, 0]),
        ('bn', [30]),
        ('max_pool2d', [2, 2, 0]),
        ('relu', [True]),

        ('SqueezeExcite', [30]),

        # inception3输入为30，与conv2 中输出的30对应
        ('branch1x1', [16, 30, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 30, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]),

        ('branch3x3_1', [16, 30, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 30, 1, 1, 1, 0]),

        # 残差项
        ('downsample', [88, 30, 1, 1, 1, 0]),  # 输入不确定；含一个conv2d 一个bn
        ('relu', [True]),
        # __________inception3结束__________
        ('flatten', []),
        ('linear', [5, 88 * 8 * 8])  # x.shape后三位参数
    ]
    loaded_model = Learner_inception_new(config_inception_Residual_se)

    # 加载保存的模型参数
    model_name = "model_path/20230916-1718.pth"
    state_dict  = torch.load(model_name)

    loaded_model.load_state_dict(state_dict, strict=False) # 加载部分参数

    # loaded_model.to(device)
    # loaded_model.eval() # 固定BN和DropOut

    ###### 修稿输出层
    # 新任务的类别数
    num_classes = 230

    # 找到线性层的位置
    last_linear_layer_index = -1  # 线性层是模型的最后一层
    config_inception_Residual_se[last_linear_layer_index] = ('linear', [num_classes,  88 * 8 * 8])

    # 线性层的参数已经被添加到 vars 中，需要更新它们
    loaded_model.vars[last_linear_layer_index * 2] = nn.Parameter(torch.ones(num_classes, 88 * 8 * 8))
    loaded_model.vars[last_linear_layer_index * 2 + 1] = nn.Parameter(torch.zeros(num_classes))


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(loaded_model.parameters(), lr=0.001, alpha=0.9)
    # 加载微调数据
    # 加载模板数据
    batch_size = 16
    raw_modeldata = np.load("F:\jupyter_notebook\DAGAN\datasets\IITDdata_right.npy",
                            allow_pickle=True).copy()  # numpy.ndarray
    model_dataloader = modeldataloader(raw_data=raw_modeldata, num_of_classes=raw_modeldata.shape[0], shuffle=True,
                                       batch_size=batch_size)
    # 微调训练循环
    timestr = time.strftime('%Y%m%d_%H%M')
    writer = SummaryWriter('finetune_logs/'+timestr) # tensorboard

    for epoch in range(20):
        print("---------",epoch)
        corrects = 0
        len = 0
        for batch_idx, (images, labels) in enumerate(model_dataloader):
            optimizer.zero_grad()
            outputs = loaded_model(images)
            pred = F.softmax(outputs, dim=1).argmax(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            corrects += torch.eq(pred, labels).sum().item()  # 可以考虑改进F1 score
            writer.add_scalar('finetune/loss',loss.item(), epoch*model_dataloader.__len__()+batch_idx)

        accs = corrects / raw_modeldata.shape[0]  # 可以考虑改进F1 score
        print("accs:", accs)
        writer.add_scalar('finetune/acc', accs, epoch)

    torch.save(loaded_model.state_dict(), "F:\jupyter_notebook\MAML-Palm\model_path/finetunemodel/"+time.strftime("%Y%m%d-%H%M",time.localtime())+".pth")

