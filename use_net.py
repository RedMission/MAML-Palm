import torch
from meta import Meta # 网络
from learner import Learner

if __name__ == '__main__':
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
        ('linear', [10, 32 * 5 * 5])
    ]

    # 加载训练好的模型
    model_name = "20230831-2025.pth"
    # 创建一个与保存的模型结构相同的实例
    loaded_model = Learner(config)  # 项目原始网络
    # 加载保存的模型参数
    loaded_model.load_state_dict(torch.load("model_path/" + model_name))
    loaded_model.to('cuda')
    loaded_model.eval()

    z = torch.randn((1,3,84,84)).to('cuda')
    logits = loaded_model(z, vars=None, bn_training=True)
    print(logits)