import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta # 网络

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

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
        ('linear', [args.n_way, 32 * 5 * 5])
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
        ('conv2d', [20, 88, 5, 5, 1, 0]),
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
        ('linear', [args.n_way, 88*18*18]) # x.shape后三位参数
    ]
    config_inception3 = [  # 参数按照网上的例子来的 格式：[('类型',[参数，参数，…]),()]
        ('conv2d', [10, 3, 5, 5, 1, 0]),
        ('relu', [True]),
        ('bn', [10]),
        ('max_pool2d', [2, 2, 0]),

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
        ('conv2d', [20, 88, 5, 5, 1, 0]),
        ('relu', [True]),
        ('bn', [20]),
        ('max_pool2d', [2, 2, 0]),

        # inception2 输入为20，与conv2 中输出的20对应
        ('branch1x1', [16, 20, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 20, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]),

        ('branch3x3_1', [16, 20, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 20, 1, 1, 1, 0]),
        # __________inception2结束__________
        # ————————下一个卷积层————————
        ('conv2d', [30, 88, 5, 5, 1, 0]),
        ('relu', [True]),
        ('bn', [30]),
        ('max_pool2d', [2, 2, 0]),

        # inception3输入为30，与conv2 中输出的30对应
        ('branch1x1', [16, 30, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 30, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]),

        ('branch3x3_1', [16, 30, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 30, 1, 1, 1, 0]),
        # __________inception3结束__________

        ('flatten', []),
        ('linear', [args.n_way, 88 * 7 * 7])  # x.shape后三位参数
    ]
    config_inception2 = [  # 参数按照网上的例子来的 格式：[('类型',[参数，参数，…]),()]
        ('conv2d', [20, 3, 5, 5, 1, 0]),
        ('relu', [True]),
        ('bn', [20]),
        ('max_pool2d', [2, 2, 0]),

        # inception 输入为10，与conv1 中的10对应
        ('branch1x1', [16, 20, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 20, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]),

        ('branch3x3_1', [16, 20, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 20, 1, 1, 1, 0]),

        # inception1结束
        # ————————下一个卷积层————————
        ('conv2d', [30, 88, 5, 5, 1, 0]),
        ('relu', [True]),
        ('bn', [30]),
        ('max_pool2d', [2, 2, 0]),

        # inception2 输入为20，与conv2 中输出的20对应
        ('branch1x1', [16, 30, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 30, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]),

        ('branch3x3_1', [16, 30, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 30, 1, 1, 1, 0]),
        # __________inception2结束__________

        ('flatten', []),
        ('linear', [args.n_way, 88 * 18 * 18])  # x.shape后三位参数
    ]
    device = torch.device('cuda')
    maml = Meta(args, config_inception3).to(device) # 传入网络参数构建 maml网络

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print("maml:",maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode（一次选择support set和query set类别的过程） number
    mini = MiniImagenet('E:\Documents\Matlab_work\DataBase\IITD Palmprint V1\Segmented/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('E:\Documents\Matlab_work\DataBase\IITD Palmprint V1\Segmented/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    for epoch in range(args.epoch//10000):  # 不断取任务喂到maml中 得到精度
        print("epoch:",epoch)
        # fetch meta_batchsz num of episode each time
        # db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 训练数据 batchsz=10000

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db): # 取到一个任务

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry) # 返回精度，会更新内层参数

            if step % 100 == 0:
                print('step:', step, '\t training acc:', accs)

            if step % 500 == 0:  # evaluation
                # db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True) # 测试数据

                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry) # 在测试中单独调用finetuning 返回精度
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16) # 求测试数据的均值
                print('Test acc:', accs)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)

    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3) # default=1
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=2) # 原15

    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84) # 图像尺寸
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    # argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-5)

    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    # argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-9)

    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
