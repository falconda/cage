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


def update_plan(plan, valid_data, num_target, num_weapon, level):
    plan = plan[0]
    # 0:每个阶段都能使用   1：间隔1个时间段能使用
    weapon_cooldown = valid_data.weapon_cooldown.squeeze(0, 1).to(device)
    # 每个目标的可被执行时间 -1:都能执行；0：第一阶段能执行；1：第二阶段能执行
    execution_time = valid_data.execution_time.squeeze(0, 1).to(device)
    pij = valid_data.Pij.squeeze(0, 1).to(device)
    qij = valid_data.target_pij_update.squeeze(0, 1).to(device)
    vt = valid_data.Threat.squeeze(0, 1).to(device)
    sum_vt_before = sum(vt)
    T_to_A = valid_data.T_to_A.squeeze(0, 1).to(device)
    useless_weapon = []
    useless_target = []
    for weapon, target in enumerate(plan):
        # 若目标不可执行
        if execution_time[target].item() == 1:
            plan[weapon] = -1
        # 处理武器冷却时间
        elif weapon_cooldown[weapon].item() == 1 and plan[weapon].item() != -1:
            useless_weapon.append(weapon)
        # 若目标可执行且目标被打掉
        elif random.randint(0, 1) <= pij[weapon][target] and plan[weapon].item() != -1:
            useless_target.append(target)
    for target, item in enumerate(execution_time):
        # 仅第一阶段可执行的目标
        if item == 0:
            if target not in useless_target:
                useless_target.append(torch.tensor(target).to(device))
        # 仅第二阶段可执行的目标
        if item == 1:
            execution_time[target] = 0

    num_target = num_target - len(useless_target)
    num_weapon = num_weapon - len(useless_weapon)
    use_less_indices_target = [int(i) for i in useless_target]
    use_less_indices_weapon = [int(i) for i in useless_weapon]
    if len(useless_target) != 0:
        mask_vt = torch.ones(vt.size(0), dtype=torch.bool)
        mask_T_to_A = torch.ones(T_to_A.size(0), dtype=torch.bool)
        mask_qij = torch.ones(qij.size(0), dtype=torch.bool)
        mask_execution_time = torch.ones(execution_time.size(0), dtype=torch.bool)
        mask_pij = torch.ones(pij.size(1), dtype=torch.bool)

        mask_vt[use_less_indices_target] = False
        mask_T_to_A[use_less_indices_target] = False
        mask_qij[use_less_indices_target] = False
        mask_execution_time[use_less_indices_target] = False
        mask_pij[use_less_indices_target] = False

        vt = vt[mask_vt]
        T_to_A = T_to_A[mask_T_to_A]
        qij = qij[mask_qij]
        execution_time = execution_time[mask_execution_time]
        pij = pij[:, mask_pij]
    if len(useless_weapon) != 0:
        mask_pij = torch.ones(pij.size(0), dtype=torch.bool)
        mask_weapon_cooldown = torch.ones(weapon_cooldown.size(0), dtype=torch.bool)

        mask_pij[use_less_indices_weapon] = False
        mask_weapon_cooldown[use_less_indices_weapon] = False

        pij = pij[mask_pij]
        weapon_cooldown = weapon_cooldown[mask_weapon_cooldown]

    sum_vt_after = sum(vt)
    valid_data.weapon_cooldown = weapon_cooldown.unsqueeze(0).unsqueeze(0)
    valid_data.execution_time = execution_time.unsqueeze(0).unsqueeze(0)
    valid_data.Pij = pij.unsqueeze(0).unsqueeze(0)
    valid_data.Qjk = qij.unsqueeze(0).unsqueeze(0)
    valid_data.Threat = vt.unsqueeze(0).unsqueeze(0)
    valid_data.T_to_A = T_to_A.unsqueeze(0).unsqueeze(0)

    sum_v = sum(valid_data.norm_value)
    if level == 2:
        for target,base in enumerate(valid_data.T_to_A[0][0]):
            target_tensor = torch.tensor(target, dtype=torch.long).to(device)
            if valid_data.norm_value[base].item() != 0:
                valid_data.norm_value[base] -= valid_data.Qjk[0][0][target_tensor].item() * valid_data.norm_value[base].item()
    delete_vt = sum(valid_data.norm_value)

    return plan, valid_data, num_weapon, num_target, delete_vt/sum_v


def calculate_reward(plan, valid_data, num_weapon, num_target):
    plan = plan[0].cpu().numpy()
    # plan = [16, 19,  3,  0, 11,  1, 18, 13,  6, 10, 17,  9,  2, 12,  4, 14,  8,  5, 15,  7]
    # 每个目标的可被执行时间 -1:都能执行；0：第一阶段能执行；1：第二阶段能执行
    execution_time = valid_data.execution_time.squeeze(0, 1).cpu().numpy()
    weapon_cool = valid_data.weapon_cooldown.squeeze(0, 1).cpu().numpy()
    pij = valid_data.Pij.squeeze(0, 1).cpu().numpy()
    qij = valid_data.Qjk.squeeze(0, 1).cpu().numpy()
    vt = valid_data.Threat.squeeze(0, 1).cpu().numpy()
    va = valid_data.norm_value.cpu().numpy()
    Tar_to_Asset = valid_data.T_to_A.cpu().numpy()[0]
    if Tar_to_Asset.ndim == 2 and Tar_to_Asset.shape[0] == 1:
        Tar_to_Asset = Tar_to_Asset.flatten()
    qij_f = np.zeros((num_target, 3))
    if isinstance(qij, np.ndarray) and qij.ndim == 1:
        for i, j in enumerate(Tar_to_Asset):
                qij_f[i][j] = qij[i].item()
        qij = qij_f

    Qjk_start = 1 * qij
    Sur_Pij = -1 * pij + 1
    # Tar_to_Asset = np.nonzero(qij)[1]
    Pij_start = np.ones_like(pij)
    sum_Threat = sum(vt)

    for weapon, target in enumerate(plan):
        # 若目标不可执行
        if execution_time[target] == 1:
            plan[weapon] = -1
    # 计算消除的威胁值
    if Qjk_start.ndim == 1:
        Qjk_start = Qjk_start.reshape(1, -1)

    for i, temp1 in enumerate(plan):
        if temp1 >= 0:
            Pij_start[i][temp1] = Sur_Pij[i][temp1]
        if 0 <= temp1 < len(vt):
            temp = Tar_to_Asset[temp1]
            Qjk_start[temp1][temp] = Sur_Pij[i][temp1] * Qjk_start[temp1][temp]
    for target_index, item in enumerate(execution_time):
        if item == 1:
            vt[target_index] = 0
    Pij_end = np.prod(Pij_start, axis=0)
    T_sum = ((-1 * Pij_end + 1) * vt).sum()
    Qjk_new = -1 * Qjk_start + 1
    Q_end = np.prod(Qjk_new, axis=0)
    V_sum = (Q_end * va).sum()

    # 新加的前瞻性目标函数
    f3 = 0
    pij_sur = np.ones((1, len(vt)))
    fij_next = np.ones((len(plan), len(vt)))
    for weapon, target in enumerate(plan):
        if target != -1:
            pij_sur[0][target] *= (1 - pij[weapon][target])
            if weapon_cool[weapon] == 1:
                fij_next[weapon, :] = 0
            if execution_time[target] == 1:
                pij_sur[0][target] = 1
    for target in range(num_target):
        if execution_time[target] == 0:
            fij_next[:, target] = 0
    a1 = 0
    a2 = 0
    for target in range(num_target):
        a1 += vt[target] * pij_sur[0][target]
        a3 = 0
        m = 0
        for weapon in range(len(plan)):
            a3 += pij[weapon][target] * fij_next[weapon][target]
            m += fij_next[weapon][target]
        if m != 0:
            a2 += vt[target] * pij_sur[0][target] * (a3 / m)
        else:
            a2 = 0
    if a1 != 0:
        f3 = (1 / a1) * a2
    else:
        f3 = 1

    # return T_sum / sum_Threat + V_sum / sum(va)
    return T_sum, sum(va)-V_sum, sum_Threat, sum(va)

def db(num):
    # 模拟参数输入
    args = argparse.Namespace()
    args.num_weapon = num
    args.num_target = num
    args.hidden_size = 256
    args.num_layers = 1
    args.dropout = 0.1
    args.actor_lr = 5e-4
    args.critic_lr = 5e-4
    args.max_grad_norm = 2.0
    args.batch_size = 1
    args.train_size = 5
    args.test_size = 1
    args.seed = 4
    args.checkpoint = None
    args.test = True  # 设置为测试模式

    # 创建训练数据和验证数据
    valid_data = ProWTADataset(args.num_weapon, args.num_target, args.test_size, args.seed + 1)

    # 初始化模型
    actor = DRL4TSP(4, 4, args.hidden_size, args.num_weapon, args.num_target, None, wta.wta_update_mask,
                    args.num_layers, args.dropout)
    actor.to(device)

    # 加载预训练模型
    weight = torch.load(os.path.join(WTAModelPath, "actor_db.pt"), map_location='cpu')
    actor.load_state_dict(weight)
    actor = actor.to(device)

    # 进行测试
    test_loader = torch.utils.data.DataLoader(valid_data, args.batch_size, False, num_workers=0)
    test_dir = 'test'
    plan = validate(test_loader, actor, wta.compute_batch, args.num_weapon, args.num_target, None, test_dir,
                          num_plot=5)
    T_delete_0, Base_damage_0, sum_Threat, sum_b = calculate_reward(plan, valid_data, args.num_weapon, args.num_target)
    print(f'T_delete_0_db: {T_delete_0}')
    plan1, valid_data, num_weapon, num_target, delete_vt_0 = update_plan(plan, valid_data, args.num_target, args.num_weapon, 1)

    reward2 = 0
    T_delete_1 = 0
    Base_damage_1 = 0
    if num_weapon != 0 and num_target != 0:
        # 初始化模型
        actor = DRL4TSP(4, 4, args.hidden_size, num_weapon, num_target, None, wta.wta_update_mask,
                        args.num_layers, args.dropout)
        actor.to(device)
        # 加载预训练模型
        weight = torch.load(os.path.join(WTAModelPath, "actor_db.pt"), map_location='cpu')
        actor.load_state_dict(weight)
        actor = actor.to(device)
        test_loader = torch.utils.data.DataLoader(valid_data, args.batch_size, False, num_workers=0)
        plan2 = validate(test_loader, actor, wta.compute_batch, num_weapon, num_target, None, test_dir,
                               num_plot=5)
        T_delete_1, Base_damage_1,q,a = calculate_reward(plan2, valid_data, num_weapon, num_target)
        print(f'T_delete_1_db: {T_delete_1}')
        a, b, c, d, delete_vt_1 = update_plan(plan2, valid_data, args.num_target,
                                                                             args.num_weapon, 2)

    return (T_delete_1+T_delete_0)/sum_Threat,(Base_damage_0+Base_damage_1)/sum_b

# 设置测试参数
def test_model():
    # 模拟参数输入
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
    args.train_size = 5
    args.test_size = 1
    args.seed = 11
    args.checkpoint = None
    args.test = True  # 设置为测试模式

    seed = int(time.time())
    random.seed(seed)

    reward_20_0 = []
    reward_20_1 = []
    reward_20_2 = []

    reward_db_0 = []
    reward_db_1 = []
    reward_db_2 = []

    for b in range(1):
        # 创建训练数据和验证数据
        valid_data = ProWTADataset(args.num_weapon, args.num_target, args.test_size, args.seed + 1)

        # 初始化模型
        actor = DRL4TSP(4, 4, args.hidden_size, args.num_weapon, args.num_target, None, wta.wta_update_mask,
                        args.num_layers, args.dropout)
        actor.to(device)

        # 加载预训练模型
        weight = torch.load(os.path.join(WTAModelPath, "actor_db.pt"), map_location='cpu')
        actor.load_state_dict(weight)
        actor = actor.to(device)

        # 进行测试
        test_loader = torch.utils.data.DataLoader(valid_data, args.batch_size, False, num_workers=0)
        test_dir = 'test'
        plan = validate(test_loader, actor, wta.compute_batch, args.num_weapon, args.num_target, None, test_dir,
                        num_plot=5)
        T_delete_0, Base_damage_0, sum_Threat, sum_b = calculate_reward(plan, valid_data, args.num_weapon,
                                                                        args.num_target)
        print(f'T_delete_0_db: {T_delete_0}')
        plan1, valid_data, num_weapon, num_target, delete_vt_0 = update_plan(plan, valid_data, args.num_target,
                                                                             args.num_weapon, 1)

        reward2 = 0
        delete_vt_1 = 0
        T_delete_1 = 0
        if num_weapon != 0 and num_target != 0:
            # 初始化模型
            actor = DRL4TSP(4, 4, args.hidden_size, num_weapon, num_target, None, wta.wta_update_mask,
                            args.num_layers, args.dropout)
            actor.to(device)
            # 加载预训练模型
            weight = torch.load(os.path.join(WTAModelPath, "actor_db.pt"), map_location='cpu')
            actor.load_state_dict(weight)
            actor = actor.to(device)
            test_loader = torch.utils.data.DataLoader(valid_data, args.batch_size, False, num_workers=0)
            plan2 = validate(test_loader, actor, wta.compute_batch, num_weapon, num_target, None, test_dir,
                             num_plot=5)
            T_delete_1, Base_damage_1, q, a = calculate_reward(plan2, valid_data, num_weapon, num_target)
            print(f'T_delete_1_db: {T_delete_1}')
            a, b, c, d, delete_vt_1 = update_plan(plan2, valid_data, args.num_target,
                                                  args.num_weapon, 2)
            print(f'base_save_db: {delete_vt_1}')

        reward_db_0.append((T_delete_1 + T_delete_0) / sum_Threat)
        print((T_delete_1 + T_delete_0) / sum_Threat)
        reward_db_1.append(delete_vt_1 / sum_b)

        # 创建训练数据和验证数据
        valid_data = ProWTADataset(args.num_weapon, args.num_target, args.test_size, args.seed + 1)

        # 初始化模型
        actor = DRL4TSP(4, 4, args.hidden_size, args.num_weapon, args.num_target, None, wta.wta_update_mask,
                        args.num_layers, args.dropout)
        actor.to(device)

        # 加载预训练模型
        weight = torch.load(os.path.join(WTAModelPath, "actor_20.pt"), map_location='cpu')
        actor.load_state_dict(weight)
        actor = actor.to(device)

        # 进行测试
        test_loader = torch.utils.data.DataLoader(valid_data, args.batch_size, False, num_workers=0)
        test_dir = 'test'
        plan = validate(test_loader, actor, wta.compute_batch, args.num_weapon, args.num_target, None, test_dir,
                              num_plot=5)
        T_delete_0, Base_damage_0, sum_Threat, sum_b = calculate_reward(plan, valid_data, args.num_weapon, args.num_target)
        print(f'T_delete_0: {T_delete_0}')
        plan1, valid_data, num_weapon, num_target, delete_vt_0 = update_plan(plan, valid_data, args.num_target, args.num_weapon, 1)

        reward2 = 0
        delete_vt_1 = 0
        T_delete_1 = 0
        if num_weapon != 0 and num_target != 0:
            # 初始化模型
            actor = DRL4TSP(4, 4, args.hidden_size, num_weapon, num_target, None, wta.wta_update_mask,
                            args.num_layers, args.dropout)
            actor.to(device)
            # 加载预训练模型
            weight = torch.load(os.path.join(WTAModelPath, "actor_20.pt"), map_location='cpu')
            actor.load_state_dict(weight)
            actor = actor.to(device)
            test_loader = torch.utils.data.DataLoader(valid_data, args.batch_size, False, num_workers=0)
            plan2 = validate(test_loader, actor, wta.compute_batch, num_weapon, num_target, None, test_dir,
                                   num_plot=5)
            T_delete_1, Base_damage_1,q,a = calculate_reward(plan2, valid_data, num_weapon, num_target)
            print(f'T_delete_1: {T_delete_1}')
            a, b, c, d, delete_vt_1 = update_plan(plan2, valid_data, args.num_target,
                                                                               args.num_weapon, 2)
            print(f'base_save_20: {delete_vt_1}')

        reward_20_0.append((T_delete_1+T_delete_0)/sum_Threat)
        print((T_delete_1 + T_delete_0) / sum_Threat)
        reward_20_1.append(delete_vt_1)
        # reward_5_2.append(reward1 + reward2)

    reward_20_0_np = [r.detach().cpu().item() if isinstance(r, torch.Tensor) else r for r in reward_20_0]
    print(f'20_v_delete_reward: {np.mean(reward_20_0_np)}')
    reward_20_1_np = [r.detach().cpu().item() if isinstance(r, torch.Tensor) else r for r in reward_20_1]
    print(f'20_b_damage_reward: {np.mean(reward_20_1_np)}')
    print("======")
    reward_db_0_np = [r.detach().cpu().item() if isinstance(r, torch.Tensor) else r for r in reward_db_0]
    print(f'db_v_delete_reward: {np.mean(reward_db_0_np)}')
    reward_db_1_np = [r.detach().cpu().item() if isinstance(r, torch.Tensor) else r for r in reward_db_1]
    print(f'db_b_damage_reward: {np.mean(reward_db_1_np)}')














if __name__ == '__main__':
    test_model()
