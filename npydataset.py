import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import random

'''
修改为根据npy array矩阵生成元学习数据
'''
class NpyDataset(Dataset):
    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize):
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to

        # self.transform = transforms.Compose([lambda x:self.render_img(x), # lambda导致无法序列化
        #                              transforms.Resize((84, 84)),
        #                              transforms.ToTensor(),
        #                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #                              ])
        self.transform = transforms.Compose([self.render_transform,
                                             # transforms.ToPILImage(),
                                             transforms.Resize((84, 84)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                             ])

        # 读入数据集合array [class_num, pre_class, imgsize, imgsize, c]
        self.raw_data = np.load(root, allow_pickle=True).copy()
        self.cls_num = self.raw_data.shape[0]
        self.cls_sample_num = self.raw_data.shape[1]
        # if mode == 'train':
        #     self.cls_sample_num = range(self.raw_data[0]*0.8)
        # else:
        #     self.cls_sample_num = range(self.raw_data[0]*0.8,self.raw_data[0])
        self.create_batch(self.batchsz)
    def render_img(self, arr):
        '''
        处理array的功能函数
        '''
        arr = (arr * 0.5) + 0.5
        arr = np.uint8(arr * 255)

        # 转换格式为PIL.Image.Image mode='RGB'
        img = Image.fromarray(np.squeeze(arr), mode='L').convert('RGB')
        return img

    def render_transform(self, x):
        return self.render_img(x)

    def create_batch(self, batchsz):

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly 选择n个行号
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)  # 打乱行号顺序
            support_x = []
            query_x = []
            for cls in selected_cls: # 在行号下选出k_shot + k_query张图，原代码选的是文件名字 此处行信息会丢失，所以得保存
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = random.sample(range(self.cls_sample_num), self.k_shot + self.k_query)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtrain_ = [(cls,j) for j in indexDtrain]
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                indexDtest_ = [(cls,j) for j in indexDtest]

                support_x.append(indexDtrain_)  # get all array for current Dtrain
                query_x.append(indexDtest_) #[array[(i,j),(i,j),(i,j)]

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        # [setsz, 3, resize, resize] 用来放图像数据
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)

        flatten_support_x = [item  for sublist in self.support_x_batch[index] for item in sublist] # item是元胞（i，j）
        support_y = np.array([item[0]  for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [item for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([item[0] for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):  # 作用是？
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        for i,index in enumerate(flatten_support_x):
            # (0, (113, 3))
            support_x[i] = self.transform(self.raw_data[index[0]][index[1]])

        for i,index in enumerate(flatten_query_x):
            query_x[i] = self.transform(self.raw_data[index[0]][index[1]])
        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        return self.batchsz

if __name__ == '__main__':
    np.random.choice([2,9], 6, False)
