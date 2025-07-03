import time

import torch
import numpy as np
import os.path
import wta
from model import DRL4TSP  # 引入模型文件中的 DRL4TSP 类
from trainer import validate, train_tsp  # 引入训练和验证函数
from wta import ProWTADataset  # 引入 WTA 数据集类
import argparse
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 获取当前文件的目录
current_dir = os.path.dirname(__file__)
# 拼接路径，指向当前文件夹下的 "model" 文件夹
WTAModelPath = os.path.join(current_dir, "new_model")

# 设置测试参数
def test_model():
    # 模拟参数输入
    STATIC_SIZE = 7  # (x, y)
    STATIC1_SIZE = 6
    STATIC2_SIZE = 7
    args = argparse.Namespace()
    args.num_weapon = 40
    args.num_target = 40
    args.hidden_size = 256
    args.num_layers = 1
    args.dropout = 0.1
    args.actor_lr = 5e-4
    args.critic_lr = 5e-4
    args.max_grad_norm = 2.0
    args.seed = 734298
    args.batch_size = 1
    args.train_size = 1
    args.test_size = 1
    args.seed = 11
    args.checkpoint = None
    args.test = True  # 设置为测试模式

    seed = 7
    random.seed(seed)
    similar_list =[]

    for b in range(50):
        # 创建训练数据和验证数据
        valid_data = ProWTADataset(args.train_size)

        # 初始化模型
        actor = DRL4TSP(STATIC_SIZE,
                        STATIC1_SIZE,
                        STATIC2_SIZE,
                        args.hidden_size,
                        None,
                        wta.wta_update_mask,
                        args.num_layers,
                        args.dropout).to(device)
        actor.to(device)

        # 加载预训练模型
        weight = torch.load(os.path.join(WTAModelPath, "actor.pt"), map_location='cpu')
        actor.load_state_dict(weight)
        actor = actor.to(device)

        # 进行测试
        test_loader = torch.utils.data.DataLoader(valid_data, args.batch_size, False, num_workers=0)
        test_dir = 'test'
        plan, human_plan = validate(test_loader, actor, wta.compute_batch, args.num_weapon, args.num_target, None, test_dir,
                        num_plot=5)
        for batch_index, batch in enumerate(test_loader):
            static, x0, static1, static2, Pij, Threat, Qjk, V_a, num_weapon, num_target, human_plan, weapon_relation = batch
        similar = wta.similar_test(plan, human_plan, weapon_relation)

        similar_list.append(similar)
    print(f'similar=', np.mean(similar_list))


if __name__ == '__main__':
    test_model()
