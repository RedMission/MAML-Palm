# 用于更新内循环参数
import torch
import torch.nn as nn

class LSLRoptimizer(nn.Module):
    '''
    p[i] := p[i] - learning_rate * dE/dp[i]

    '''
    def __init__(self,total_num_inner_loop_steps,init_update_lr,use_learnable_learning_rates=1):
        '''
        :param total_num_inner_loop_steps:
        :param init_update_lr:
        :param use_learnable_learning_rates:
        '''
        super(LSLRoptimizer,self).__init__()
        assert init_update_lr>0.,'learning_rate必须大于0'
        self.total_num_inner_loop_steps=total_num_inner_loop_steps
        self.use_learnable_learning_rates = use_learnable_learning_rates

    def init(self,names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
                requires_grad=self.use_learnable_learning_rates)


    def update_params(self):
        pass