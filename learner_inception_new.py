import  torch
from    torch import nn
from    torch.nn import functional as F

class Learner_inception_new(nn.Module):

    def __init__(self, config):
        super(Learner_inception_new, self).__init__()
        self.config = config
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if name is 'branch1x1':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'branch5x5_1':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'branch5x5_2':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'branch3x3_1':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'branch3x3_2':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'branch3x3_3':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'branch_pool':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def forward(self, x, vars=None, bn_training=True):
        """

        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        # branch1x1=x
        # branch5x5=x
        # branch3x3=x

        for name, param in self.config:
            if name is 'branch1x1':
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                branch1x1 = F.conv2d(x, w, b, stride=param[4], padding=param[5]) # 放入网络计算
                idx += 2

            elif name is 'branch5x5_1':
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                branch5x5 = F.conv2d(x, w, b, stride=param[4], padding=param[5]) # 放入网络计算
                idx += 2
            elif name is 'branch5x5_2':
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                branch5x5 = F.conv2d(branch5x5, w, b, stride=param[4], padding=param[5]) # 放入网络计算
                idx += 2

            elif name is 'branch3x3_1':
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                branch3x3 = F.conv2d(x, w, b, stride=param[4], padding=param[5]) # 放入网络计算
                idx += 2
            elif name is 'branch3x3_2':
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                branch3x3 = F.conv2d(branch3x3, w, b, stride=param[4], padding=param[5]) # 放入网络计算
                idx += 2
            elif name is 'branch3x3_3':
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                branch3x3 = F.conv2d(branch3x3, w, b, stride=param[4], padding=param[5]) # 放入网络计算
                idx += 2

            elif name is 'branch_pool':
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
                branch_pool = F.conv2d(branch_pool, w, b, stride=param[4], padding=param[5]) # 放入网络计算
                outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
                x = torch.cat(outputs, dim=1)
                idx += 2

            elif name is 'conv2d':
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5]) # 放入网络计算
                idx += 2
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print("----x.shape:",x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])

            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])

            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                print(name)
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        return x
    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()