# 用于更新内循环参数  待处理学习率的字典！！
import torch
import torch.nn as nn

class LSLRoptimizer(nn.Module):
    '''
    p[i] := p[i] - learning_rate * dE/dp[i]

    '''
    def __init__(self,total_num_inner_loop_steps,init_update_lr,use_learnable_learning_rates=True):
        '''
        :param total_num_inner_loop_steps:
        :param init_update_lr:
        :param use_learnable_learning_rates:
        '''
        super(LSLRoptimizer,self).__init__()
        assert init_update_lr>0.,'learning_rate必须大于0'
        self.init_learning_rate = torch.ones(1) * init_update_lr
        self.total_num_inner_loop_steps=total_num_inner_loop_steps
        self.use_learnable_learning_rates = True

    def init(self,names_weights_dict):
        self.names_learning_rates_dict= nn.ParameterDict()
        for idx, (key, param) in enumerate(names_weights_dict.items()):
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps) * self.init_learning_rate,
                requires_grad=self.use_learnable_learning_rates)
        # print("-----------names_learning_rates_dict------------")
        # print(self.names_learning_rates_dict)


    def update_params(self,names_weights_dict,names_grads_wrt_params_dict,num_step):
        # return {
        #     key: names_weights_dict[key] # 传入网络权值字典
        #          - self.names_learning_rates_dict[key.replace(".", "-")][num_step]
        #          * names_grads_wrt_params_dict[key] # 梯度字典
        #     for key in names_grads_wrt_params_dict.keys() # 梯度字典
        # }
        # print("检查内循环")
        # print(names_grads_wrt_params_dict)
        # print("学习率")
        # print(tuple(self.names_learning_rates_dict.values()))

        return  list(map(lambda p: p[1] - self.init_learning_rate * p[0], zip(names_grads_wrt_params_dict, names_weights_dict)))

