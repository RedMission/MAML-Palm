import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
# from    learner import Learner # 自定义类
from    copy import deepcopy
from learner_inception_new import Learner_inception_new
import inner_loop_optimizers


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()
        self.update_lr = args.update_lr # 内部更新学习率
        self.meta_lr = args.meta_lr #外部学习率
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.current_epoch = 0
        self.total_epoch = args.epoch

        self.net = Learner_inception_new(config) #改为inception模块构成的网络
        # 外循环优化器
        self.meta_optim  = optim.RMSprop(self.net.parameters(), lr=self.meta_lr, alpha=0.9)
        # self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr) # parameters已被重写
        # 外循环 余弦
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.meta_optim, T_max=100,
                                                              eta_min=0.0001)

        # 内循环优化器
        self.inner_loop_optimizer = inner_loop_optimizers.LSLRoptimizer(total_num_inner_loop_steps=self.task_num+1,
                                                                        init_update_lr=self.update_lr,
                                                                        use_learnable_learning_rates=True)
        # 内循环优化器初始化
        self.inner_loop_optimizer.init(  # 初始化内训环的优化器 直接传入内循环网络参数字典
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.net.named_parameters()))


    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.返回一个包含用于内循环更新的参数的字典。
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        return {
            name: param
            for name, param in params
            if param.requires_grad
            # and (
            #     not self.args.enable_inner_loop_optimizable_bn_params
            #     and "norm_layer" not in name
            #     or self.args.enable_inner_loop_optimizable_bn_params
            # )
        }


    def get_per_step_loss_importance_vector(self):
            """
            生成一个维度的张量（num_inner_loop_steps），表示每一步的目标损失对优化损失的重要性。
            :return: A tensor to be used to compute the weighted average of the loss, useful for
            the MSL (Multi Step Loss) mechanism.
            """
            loss_weights = np.ones(shape=(self.update_step + 1)) * (1.0 / (self.update_step + 1))
            decay_rate = 1.0 / (self.update_step + 1) / 10
            min_value_for_non_final_losses = 0.03 / (self.update_step + 1)
            for i in range(len(loss_weights) - 1):
                curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate),
                                        min_value_for_non_final_losses)
                loss_weights[i] = curr_value

            curr_value = np.minimum(
                loss_weights[-1] + (
                            self.current_epoch * ((self.update_step + 1) - 1) * decay_rate),
                1.0 - (((self.update_step + 1) - 1) * min_value_for_non_final_losses))
            loss_weights[-1] = curr_value
            self.current_epoch += 1 # 考虑其他从外部得到epoch的方法
            return loss_weights

    def clip_grad_by_norm_(self, grad, max_norm): #
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2) # 此处2等同于网络上梯度裁剪的norm_type类型强制为浮点数
            total_norm += param_norm.item() ** 2 # 计算所有网络参数梯度范数之和
            counter += 1
        total_norm = total_norm ** (1. / 2) # 再归一化 等价于把所有网络参数放入一个向量，再对向量计算范数

        clip_coef = max_norm / (total_norm + 1e-6) # 非负处理+ 1e-6
        if clip_coef < 1: # 判断是否溢出了预设上限
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry,MSL_flag,use_second_order):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # 新建list，元素个数为update_step数量个。losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        if MSL_flag: # 默认不开启
            # without MSL
            per_step_loss_importance_vectors = np.ones(shape=(self.update_step + 1))
        else:
            # MSL
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()  # 每步损失的权重

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            # print(x_spt[i].shape) # (30,3,84,84)
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            # grad = torch.autograd.grad(loss, self.net.parameters()) # 通过损失和参数计算梯度 不保存二阶导
            grad = torch.autograd.grad(loss, self.net.parameters(),create_graph=use_second_order,allow_unused=True) # 通过损失和参数计算梯度 二阶导
            # 更新权值 做LSLR（待完成）
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            # fast_weights = self.inner_loop_optimizer.update_params(names_weights_dict=self.net.parameters(),
            #                                                          names_grads_wrt_params_dict=grad,
            #                                                          num_step=0)

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                # print("x_qry[i].shape:",x_qry[i].shape)
                # x_qry[i].shape: torch.Size([10, 3, 84, 84])
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q * per_step_loss_importance_vectors[0]


                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item() # 可以考虑改进F1 score
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q * per_step_loss_importance_vectors[1]
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step): # 在一个任务中的跟更新step
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                # 更新权值 做LSLR（待完成）
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                # fast_weights = self.inner_loop_optimizer.update_params(names_weights_dict=fast_weights,
                #                                                        names_grads_wrt_params_dict=grad,
                #                                                        num_step=k)

                # logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # # loss_q will be overwritten and just keep the loss_q on last update step.
                # loss_q = F.cross_entropy(logits_q, y_qry[i])
                # losses_q[k + 1] += loss_q * per_step_loss_importance_vectors[k+1]
                # 降低对计算资源的要求(博客 https://blog.csdn.net/wangkaidehao/article/details/105507809)
                if k < self.update_step - 1:
                    with torch.no_grad():
                        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                        loss_q = F.cross_entropy(logits_q, y_qry[i])
                        losses_q[k + 1] += loss_q * per_step_loss_importance_vectors[k + 1]
                else:
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry[i])
                    losses_q[k + 1] += loss_q * per_step_loss_importance_vectors[k + 1]

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        if MSL_flag:
            # 原始
            loss_q = losses_q[-1] / task_num
        else:
            # MSL
            loss_q = sum(losses_q) / (task_num)

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        #  CA 余弦退火学习率
        # self.scheduler.step()

        accs = np.array(corrects) / (querysz * task_num) # 可以考虑改进F1 score
        loss = np.array([i.cpu().detach() for i in losses_q]) / task_num
        # loss是一个张量的列表
        return accs,loss

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        # losses = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        # 复制网络
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters()))) #更新了网络参数

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / querysz
        # loss = np.array(loss_q) / querysz


        return accs




def main():
    pass


if __name__ == '__main__':
    main()
