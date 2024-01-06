import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
一些功能函数
'''

def cossimiliarity(a,b):
    # 计算余弦相似度【越大越好】
    dot_product = np.dot(a, b)
    norm_sample = np.linalg.norm(a)
    norm_template = np.linalg.norm(b)
    similarity = dot_product / (norm_sample * norm_template)
    return similarity

# 欧式距离
def cal_eu(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())


# 计算概率密度曲线重合的面积
def find_max_min(df1, df2):
    max = np.maximum(df1, df2)
    min = np.minimum(df1, df2)
    return max, min


# 计算函数相交面积和
def cal_iou_bin(data1, data2, bin=100):
    stadata1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))
    stadata2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
    freq1, _ = np.histogram(stadata1, bins=bin)
    freq2, _ = np.histogram(stadata2, bins=bin)
    max, min = find_max_min(freq1, freq2)
    iou = sum(min) / sum(max)
    return iou


# DET曲线

def plotDET(ledis, iledis):
    stop = max(max(ledis), max(iledis))
    start = min(min(ledis), min(iledis))
    # 取阈值
    num = 100
    thresholdlist = np.linspace(start, stop, num=num, endpoint=True, retstep=False, dtype=None)
    far = []  # x 由于使用的是匹配相似度，从大到小生成
    frr = []  # y
    for threshold_i in range(num):
        x = len([i for i in iledis if i > thresholdlist[threshold_i]]) / len(iledis)
        y = len([i for i in ledis if i < thresholdlist[threshold_i]]) / len(ledis) # 正类别漏查 DET
        # y = len([i for i in ledis if i >= thresholdlist[threshold_i]]) / len(ledis) # 正类正查 roc
        far.append(x)
        frr.append(y)
    # 求曲线交点
    # 选出两曲线被包含的区间
    points1 = [t for t in zip(far, frr) if far[0] >= t[0] >= far[-1]]
    points2 = [t for t in zip(far, far) if far[0] >= t[0] >= far[-1]]
    idx = 0
    nrof_points = len(points1)
    while idx < nrof_points - 1:
        # 将数据分段
        x3 = np.linspace(points1[idx][0], points1[idx + 1][0], 1000)
        y1_new = np.linspace(points1[idx][1], points1[idx + 1][1], 1000)
        y2_new = np.linspace(points2[idx][1], points2[idx + 1][1], 1000)

        tmp_idx = np.argwhere(np.isclose(y1_new, y2_new, atol=0.0003)).reshape(-1)
        if tmp_idx.any():
            plt.scatter(x3[tmp_idx], y2_new[tmp_idx], c='r', marker='x')
            print('EER:', x3[tmp_idx])
            # 计算阈值
            countlen = len(iledis) * x3[tmp_idx]
            threshold = sorted(iledis)[int(countlen[0])]
            print('threshold:', threshold)
        idx += 1

    plt.plot(far, frr, color='b', linestyle='-')
    # 等误率
    plt.plot([-0.2, 0.9], [-0.2, 0.9], color='y', linestyle='-.')
    plt.xlim((0, 0.9))
    plt.ylim((0, 0.9))
    plt.title('DET curve')
    plt.show()
    # return threshold

# 复合DET曲线
def plotDET_muti(value,labels):  # 输入合法值与非法值元组构成的list
    linestyle = ['dotted', 'dashed', 'dashdot', (0, (1, 1)), (0, (1, 2)), (0, (2, 1)), (0, (2, 2))]
    color = ['r', 'b', 'g', 'm', 'k', 'c']
    plt.plot([-0.2, 1], [-0.2, 1], color='y', linestyle='-.')
    for index in range(len(value)):
        print('model=', labels[index])
        ledis = value[index][0]
        iledis = value[index][1]
        stop = max(max(ledis), max(iledis))
        start = min(min(ledis), min(iledis))
        # 取阈值
        num = 50
        thresholdlist = np.linspace(start, stop, num=num, endpoint=True, retstep=False, dtype=None)
        far = []  # x 由于使用的是匹配相似度，从大到小生成
        frr = []  # y
        for threshold in range(num):
            x = len([i for i in iledis if i > thresholdlist[threshold]]) / len(iledis)
            y = len([i for i in ledis if i < thresholdlist[threshold]]) / len(ledis)
            far.append(x)
            frr.append(y)
        # 求曲线交点
        # 选出两曲线被包含的区间
        points1 = [t for t in zip(far, frr) if far[0] >= t[0] >= far[-1]]
        points2 = [t for t in zip(far, far) if far[0] >= t[0] >= far[-1]]
        idx = 0
        nrof_points = len(points1)
        while idx < nrof_points - 1:
            # 将数据分段
            x3 = np.linspace(points1[idx][0], points1[idx + 1][0], 1000)
            y1_new = np.linspace(points1[idx][1], points1[idx + 1][1], 1000)
            y2_new = np.linspace(points2[idx][1], points2[idx + 1][1], 1000)
            tmp_idx = np.argwhere(np.isclose(y1_new, y2_new, atol=0.0003)).reshape(-1)
            if tmp_idx.any():
                plt.scatter(x3[tmp_idx], y2_new[tmp_idx], c='r', marker='x', s=4)  # s调节点的大小
                print('EER:', x3[tmp_idx][0])
                # 计算阈值
                countlen = len(iledis) * x3[tmp_idx]
                threshold = sorted(iledis)[int(countlen[0])]
                print('threshold:', threshold)
            idx += 1
        plt.plot(far, frr, color=color[index % len(color)], linestyle=linestyle[index % len(linestyle)],
                 label=labels[index])
    # 等误率
    plt.xlim((0, 0.6))
    plt.ylim((0, 0.6))
    plt.title('DET curve')
    plt.xlabel('FAR')
    plt.ylabel('FRR')
    plt.legend()
    plt.show()


# 计算函数相交面积和
def cal_iou_bin(data1, data2, bin=100):
    stadata1= (data1- np.min(data1)) / (np.max(data1) - np.min(data1))
    stadata2= (data2- np.min(data2)) / (np.max(data2) - np.min(data2))
    freq1, _ = np.histogram(stadata1, bins=bin)
    freq2, _ = np.histogram(stadata2, bins=bin)
    max, min = find_max_min(freq1, freq2)
    iou = sum(min)/sum(max)
    return iou
def kdeplot_muti(value,labels):
    for index in range(len(value)):
        print('model=', labels[index])
        ledis = value[index][0]
        iledis = value[index][1]
        # 核密度函图
        # plt.figure(figsize=(8, 5), dpi=500)
        p1 = sns.kdeplot(ledis, color="g", shade=True)
        p1 = sns.kdeplot(iledis, color="r", shade=True)
        plt.xlabel(labels[index]+' 匹配相似度')
        plt.ylabel('概率密度估计值')
        # 设置 x 轴的范围为 0 到 1.2
        # plt.xlim(0, 1.3)
        plt.show()
        # 计算面积
        iou = cal_iou_bin(ledis, iledis, bin=100)
        print("iou:", iou)

