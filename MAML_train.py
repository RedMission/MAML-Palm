import  torch
import  numpy as np
import  scipy.stats
from    torch.utils.data import DataLoader
import  argparse
from meta import Meta # 网络
from torch.utils.tensorboard import SummaryWriter
from npydataset import NpyDataset


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main():

    # torch.manual_seed(222) #设置随机种子后，是每次运行文件的输出结果都一样
    # torch.cuda.manual_seed_all(222)
    # np.random.seed(222)

    torch.manual_seed(122)  # 设置随机种子后，是每次运行文件的输出结果都一样
    torch.cuda.manual_seed_all(122)
    np.random.seed(122)

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
    config_inception_Residual = [  # 格式：[('类型',[参数，参数，…]),()]
        ('conv2d', [10, 3, 3, 3, 1, 0]),
        ('bn', [10]),
        ('max_pool2d', [2, 2, 0]),
        ('relu', [True]),

        # inception 输入为10，与conv1 中的10对应
        ('branch1x1', [16, 10, 1, 1, 1, 0]),

        ('branch5x5_1', [16, 10, 1, 1, 1, 0]),
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]), # 核考虑换成3*3

        ('branch3x3_1', [16, 10, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 10, 1, 1, 1, 0]),
        # 残差项
        ('downsample', [88, 10, 1, 1, 1, 0]), # 输入不确定；含一个conv2d 一个bn

        ('relu', [True]),

        # inception1结束
        # ————————下一个卷积层————————
        ('conv2d', [20, 88, 3, 3, 1, 0]), #
        ('bn', [20]),
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
        # 残差项
        ('downsample', [88, 20, 1, 1, 1, 0]),  # 输入不确定；含一个conv2d 一个bn
        ('relu', [True]),
        # __________inception2结束__________
        # ————————下一个卷积层————————
        ('conv2d', [30, 88, 3, 3, 1, 0]),
        ('bn', [30]),
        ('max_pool2d', [2, 2, 0]),
        ('relu', [True]),

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
        ('linear', [args.n_way, 88 * 8 * 8])  # x.shape后三位参数
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
        ('branch5x5_2', [24, 16, 5, 5, 1, 2]), # 核考虑换成3*3

        ('branch3x3_1', [16, 10, 1, 1, 1, 0]),
        ('branch3x3_2', [24, 16, 3, 3, 1, 1]),
        ('branch3x3_3', [24, 24, 3, 3, 1, 1]),

        ('branch_pool', [24, 10, 1, 1, 1, 0]),
        # 残差项
        ('downsample', [88, 10, 1, 1, 1, 0]), # 输入不确定；含一个conv2d 一个bn

        ('relu', [True]),

        # inception1结束
        # ————————下一个卷积层————————
        ('conv2d', [20, 88, 3, 3, 1, 0]), #
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
        ('linear', [args.n_way, 88 * 8 * 8])  # x.shape后三位参数
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maml = Meta(args, config).to(device) # 传入网络参数构建 maml网络

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print("maml:",maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode（一次选择support set和query set类别的过程） number
    train_data = NpyDataset(root = args.train_data,
                      mode='train', n_way=args.n_way,
                        k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=args.t_batchsz,  #
                        resize=args.imgsz)
    test_data = NpyDataset(root = args.test_data, mode='test',
                             n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100,
                             resize=args.imgsz)

    writer = SummaryWriter() # tensorboard
    for epoch in range(args.epoch//args.t_batchsz):  #
        print("epoch:",epoch)
        # db = DataLoader(train_data, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        db = DataLoader(train_data, args.task_num, shuffle=True, num_workers=4, pin_memory=True) # 生成可以将所有任务跑一遍的迭代器

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db): # 从迭代器取任务组合，每组完成一次外层循环，共step步外循环
            # use_second_order在训练中是否使用二阶导
            # 前50 false
            use_second_order = False
            # if step < 700:
            #     use_second_order = False
            # else:
            #     use_second_order = True

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs,loss = maml(x_spt, y_spt, x_qry, y_qry,args.MSL_flag,use_second_order) # 传入的多个任务(共task_num个)
            # 可视化
            writer.add_scalar('train/loss',loss[-1].item(), epoch*(args.t_batchsz//args.task_num)+step)
            writer.add_scalar('train/acc',accs[-1].item(), epoch*(args.t_batchsz//args.task_num)+step)

            if step % 100 == 0:
                print('step:', step, '\t training acc:', accs)

            if step % 200 == 0:  # evaluation
                # db_test = DataLoader(test_data, 1, shuffle=True, num_workers=1, pin_memory=True)
                db_test = DataLoader(test_data, 1, shuffle=True, num_workers=4, pin_memory=True) # 测试 生成可以将所有任务跑一遍的迭代器
                accs_all_test = []
                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    test_accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry) # 在测试中单独调用finetuning 返回精度
                    accs_all_test.append(test_accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16) # 求测试数据的均值
                writer.add_scalar('test/acc', accs[-1].item(), epoch * (args.t_batchsz // args.task_num) + step)
                print('Test acc:', accs)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_data', type=str, help='', default='F:\jupyter_notebook\DAGAN\datasets\IITDdata_left_6.npy')
    argparser.add_argument('--test_data', type=str, help='', default='F:\jupyter_notebook\DAGAN\datasets\IITDdata_right.npy')

    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)

    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3) # default=1
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=2) # 原15
    argparser.add_argument('--t_batchsz', type=int, help='train-batchsz', default=5000)

    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84) # 调节的图像尺寸
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    # argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-5)

    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    # argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-9)

    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--MSL_flag', type=bool, help='是否使用多步损失', default=False)

    args = argparser.parse_args()

    main()
