import  torch
from    torch import nn
from    torch.nn import functional as F


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Learner_inception_new(nn.Module):
    def __init__(self, config):
        super(Learner_inception_new, self).__init__()
        self.config = config # 获取参数列表
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList() # 新建
        # running_mean and running_var
        self.vars_bn = nn.ParameterList() # 新建
        for i, (name, param) in enumerate(self.config): # 获取列表元组
            if name in ['branch1x1','branch5x5_1','branch5x5_2','branch3x3_1','branch3x3_2',
                          'branch3x3_3','branch_pool','conv2d','convt2d','linear']:
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'downsample':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                # bn
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name is 'SqueezeExcite':
                in_chs = param[0]
                se_ratio = 0.25
                divisor = 4
                reduced_chs = _make_divisible((in_chs) * se_ratio, divisor)
                conv_reduce_config = [reduced_chs, in_chs, 1, 1]
                w = nn.Parameter(torch.ones(conv_reduce_config))
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                self.vars.append(nn.Parameter(torch.zeros(reduced_chs)))

                conv_expand_config=[in_chs, reduced_chs, 1, 1]
                w = nn.Parameter(torch.ones(conv_expand_config))
                torch.nn.init.kaiming_normal_(w)  # 使用正态分布对输入张量进行赋值
                self.vars.append(w)  # 加入list
                self.vars.append(nn.Parameter(torch.zeros(in_chs)))

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
                branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                branch_pool = F.conv2d(branch_pool, w, b, stride=param[4], padding=param[5]) # 放入网络计算
                outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
                inception_x = torch.cat(outputs, dim=1)
                idx += 2
            elif name is 'downsample':
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                residual = F.conv2d(x, w, b, stride=param[4], padding=param[5]) # 输入考虑一下原始x
                idx += 2
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                residual = F.batch_norm(residual, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

                x = inception_x + residual
            elif name is 'SqueezeExcite':
                x_se = nn.AdaptiveAvgPool2d(1)(x)
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                x_se = F.conv2d(x_se, w, b, stride=1, padding=0) # 放入网络计算
                idx += 2
                x_se = F.relu(x_se, inplace=True)
                w, b = vars[idx], vars[idx + 1] #取出权重和偏置
                x_se = F.conv2d(x_se, w, b, stride=1, padding=0) # 放入网络计算
                idx += 2
                x_se = F.relu6(x_se + 3.) / 6.
                x = x * x_se

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