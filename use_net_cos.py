import time

import nmslib
import numpy as np
import torch
from  torch import nn

import function
from learner_inception_new import Learner_inception_new
from dataloader import modeldataloader, normaldataloader
from    torch.nn import functional as F
import faiss

'''
用网络提取特征 计算距离  1232/1380  0.89 平均时间:3.67→0.39
'''

def match(unknowdata, model):
    return


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
        # ('linear', [230, 88 * 8 * 8])  # x.shape后三位参数
    ]
    loaded_model = Learner_inception_new(config_inception_Residual_se)

    # 加载保存的模型参数
    model_name = "model_path/new_config/IITD/20230904-1239.pth"
    state_dict  = torch.load(model_name)

    loaded_model.load_state_dict(state_dict, strict=False) # 加载部分参数
    loaded_model.to(device)
    # loaded_model.eval()

    # 加载模板数据
    raw_modeldata = np.load("F:\jupyter_notebook\DAGAN\datasets\IITDdata_right.npy", allow_pickle=True).copy() #numpy.ndarray
    modeldataloader = modeldataloader(raw_data=raw_modeldata, num_of_classes=raw_modeldata.shape[0], shuffle=False, batch_size=1)
    unknowdataloader = normaldataloader(raw_data=raw_modeldata, num_of_classes=raw_modeldata.shape[0], shuffle=True, batch_size=1)
    # 计算模板
    model = []
    for i,item  in enumerate(modeldataloader): # 每个类别下随机取一张作为注册模板向量
        model_data, model_label = item
        model_i = loaded_model(model_data.to(device), vars=None, bn_training=True)
        model.append(model_i.detach().numpy().reshape(-1))

    # 创建搜索算法需要的index
    index_data = np.array(model, dtype='float32')

    # nmslib 初始化搜索算法index, 使用HNSW、Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(index_data)
    index.createIndex({'post': 2}, print_progress=True)

     # faiss初始化搜索算法index
    dim, measure = index_data.shape[1], faiss.METRIC_INNER_PRODUCT # 内积
    param = 'PCA32,HNSW32'
    index_faiss = faiss.index_factory(dim, param, measure)
    index_faiss.train(index_data) # 加pca后需要先训练
    index_faiss.add(index_data)

    count = 0
    count_nms = 0
    count_faiss = 0

    ledis, iledis = [],[]
    count_time = 0
    count_nmstime = 0
    count_faisstime = 0

    for i,item  in enumerate(unknowdataloader):
        unknow_data, unknow_label = item
        # 计算特征
        vector = loaded_model(unknow_data.to(device), vars=None, bn_training=True).detach().numpy().reshape(-1)
        print("-------")
        print("真实:",unknow_label.item())
        # 匹配相似度列表
        tmp = []
        T1 = time.clock()
        # 与模板逐个匹配
        for i in range(len(model)):
            unknow_vector = function.cossimiliarity(vector,model[i]) # 此处计算的是匹配度，越大越可能是同一个体
            tmp.append(unknow_vector)
            if i == unknow_label.item():
                ledis.append(unknow_vector) # 合法匹配
            else:
                iledis.append(unknow_vector) # 非法匹配

        log = tmp.index(max(tmp))
        T2 = time.clock()
        count_time += (T2 - T1)

        T3 = time.clock()
        ids, distances = index.knnQuery(vector, k=5)
        print(ids)
        T4 = time.clock()
        count_nmstime += (T4 - T3)

        T5 = time.clock()
        Dis, Ind = index_faiss.search(np.expand_dims(vector, axis=0), k=1) # vector需要修改维度
        print(Ind)
        T6 = time.clock()
        count_faisstime += (T6 - T5)


        print("逐个比对的预测:",log)
        if unknow_label.item() == log:
            count += 1
        print("向量数据库的预测:", ids)
        if unknow_label.item() == ids[0]:
            count_nms += 1
        print("faiss向量数据库的预测:", Ind)
        if unknow_label.item() == Ind[0][0]:
            count_faiss += 1

    print("----------")
    print("逐个比对正确预测数量:",count,"acc:",count/len(unknowdataloader))
    print("向量数据库比对正确预测数量:",count_nms,"acc:",count_nms/len(unknowdataloader))
    print("faiss向量数据库比对正确预测数量:",count_faiss,"acc:",count_faiss/len(unknowdataloader))

    print("逐个比对平均时间:%s毫秒"%((count_time*1000)/len(unknowdataloader)))
    print("向量数据库比对平均时间:%s毫秒"%((count_nmstime*1000)/len(unknowdataloader)))
    print("faiss向量数据库比对平均时间:%s毫秒"%((count_faisstime*1000)/len(unknowdataloader)))

    # function.plotDET(ledis, iledis)



