# 用于更新内循环参数
import torch
import torch.nn as nn

class LSLRoptimizer(nn.Module):
    '''
    p[i] := p[i] - learning_rate * dE/dp[i]
    '''
    def __init__(self,grad,parameters,init_update_lr,):
        super(LSLRoptimizer,self).__init__()
        assert init_update_lr>0.,'learning_rate必须大于0'
        self.

    def init(self):

    def update_params(self):