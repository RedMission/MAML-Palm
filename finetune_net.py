import time

import numpy as np
import torch
from  torch import nn
from torch.utils.tensorboard import SummaryWriter

from learner_inception_new import Learner_inception_new
from dataloader import modeldataloader, normaldataloader
from    torch.nn import functional as F

def match(unknowdata, model):
    return

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
    model_name = "model_path/new_config/20230904-1239.pth"
    state_dict  = torch.load(model_name)

    loaded_model.load_state_dict(state_dict, strict=False) # 加载部分参数
    # print(loaded_model.vars)
    # loaded_model.to(device)
    # loaded_model.eval()
    '''


    # 加载模板数据
    raw_modeldata = np.load("F:\jupyter_notebook\DAGAN\datasets\IITDdata_right.npy", allow_pickle=True).copy() #numpy.ndarray
    model_dataloader = modeldataloader(raw_data=raw_modeldata, num_of_classes=raw_modeldata.shape[0], shuffle=False, batch_size=1)
    计算模板
    model = []
    for i,item  in enumerate(model_dataloader): # 每个类别下随机取一张作为注册模板向量
        model_data, model_label = item
        model_i = loaded_model(model_data.to(device), vars=None, bn_training=True)
        model.append(model_i.detach().numpy().reshape(-1))

    # 加载待检测数据
    raw_unknowdata = np.load("F:\jupyter_notebook\DAGAN\datasets\IITDdata_right.npy",
                       allow_pickle=True).copy()  # numpy.ndarray
    unknow_dataloader = normaldataloader(raw_data=raw_unknowdata, num_of_classes=200, shuffle=True,
                                         batch_size=1)
    count = 0
    for i,item  in enumerate(unknow_dataloader):
        unknow_data, unknow_label = item
        vector = loaded_model(unknow_data.to(device), vars=None, bn_training=True)
        unknow_data, unknow_label=vector.detach().numpy().reshape(-1), unknow_label.item()
        print("-------")
        print("真实:",unknow_label)
        tmp = []
        for model_i in model:
            tmp.append(cossimiliarity(unknow_data,model_i))
        log = tmp.index(max(tmp))
        print("预测:",log)
        if unknow_label == log:
            count += 1
    print(count)

    '''
    #
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
    optimizer = torch.optim.RMSprop(loaded_model.parameters(), lr=0.1, alpha=0.9)
    # 加载微调数据
    # 加载模板数据
    raw_modeldata = np.load("F:\jupyter_notebook\DAGAN\datasets\IITDdata_right.npy",
                            allow_pickle=True).copy()  # numpy.ndarray
    model_dataloader = modeldataloader(raw_data=raw_modeldata, num_of_classes=raw_modeldata.shape[0], shuffle=True,
                                       batch_size=16)
    # 微调训练循环
    timestr = time.strftime('%Y%m%d_%H%M')
    writer = SummaryWriter('finetune_logs/'+timestr) # tensorboard

    for epoch in range(500):
        print("---------",epoch)
        for batch_idx, (images, labels) in enumerate(model_dataloader):
            optimizer.zero_grad()
            outputs = loaded_model(images)
            print("outputs:",outputs)
            pred = F.softmax(outputs, dim=1).argmax(dim=1)
            print("pred:",pred)
            loss = criterion(outputs, labels)
            aaa = criterion(pred, labels)
            print(loss.item())
            writer.add_scalar('finetune-train/loss',loss.item(), epoch*model_dataloader.__len__()+batch_idx)
            loss.backward()
            optimizer.step()
    torch.save(loaded_model.state_dict(), "F:\jupyter_notebook\MAML-Palm\model_path/"+time.strftime("%Y%m%d-%H%M",time.localtime())+".pth")

