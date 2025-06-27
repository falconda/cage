import numpy as np
import random
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence
import matplotlib
import torch.nn.functional as F
import pandas as pd
import ast

matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = '数据库.xlsx'  # 使用相对路径
data = pd.read_excel(file_path)
line = 0


def wta_update_mask(mask, chosen_idx, num_weapon=20, num_target=20):
    weapon_index = torch.div(chosen_idx, num_target, rounding_mode='floor')
    start_a = torch.mul(weapon_index, num_target)
    end_a = torch.mul(torch.add(weapon_index, 1), num_target)
    weapon_array = []
    for i in range(chosen_idx.size()[0]):
        weapon_array.append(list(range(int(start_a[i]), int(end_a[i]), 1)))
    weapon_mask = torch.tensor(weapon_array).to(device)
    target_array = []
    target_index = torch.remainder(chosen_idx, num_target)
    end_c = torch.add(target_index, num_weapon * num_target)
    for i in range(chosen_idx.size()[0]):
        target_array.append(list(range(target_index[i], end_c[i], num_target)))
    target_mask = torch.tensor(target_array).to(device)
    mask.scatter_(1, weapon_mask, 0)
    mask.scatter_(1, target_mask, 0)
    return mask


class ProWTADataset(Dataset):

    def __init__(self, num_samples=1):
        super(ProWTADataset, self).__init__()

        seed = np.random.randint(123456789)
        global line
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        # 按行遍历前num_samples行
        for index, row in data.iloc[line:num_samples+line].iterrows():
            line += 1
            # 提取蓝方的数据
            self.target_type = (torch.tensor(ast.literal_eval(row['目标类型'])))
            self.target_coordinates = (torch.tensor(ast.literal_eval(row['目标经纬度'])))
            self.target_speed = (torch.tensor(ast.literal_eval(row['目标速度'])))
            self.target_azimuth = (torch.tensor(ast.literal_eval(row['目标方位角'])))
            self.target_range = (torch.tensor(ast.literal_eval(row['作战范围'])))
            # 提取红方的数据
            self.weapon_type = (torch.tensor(ast.literal_eval(row['红方武器类型'])))
            self.weapon_coordinates = (torch.tensor(ast.literal_eval(row['红方武器经纬度'])))
            self.weapon_damage = (torch.tensor(ast.literal_eval(row['武器毁伤程度'])))
            self.weapon_range = (torch.tensor(ast.literal_eval(row['打击范围'])))
            self.weapon_ammunition = (torch.tensor(ast.literal_eval(row['弹药类型'])))
            self.weapon_weight = (torch.tensor(ast.literal_eval(row['武器部火药重量'])))
            self.weapon_relation = (torch.tensor(ast.literal_eval(row['拦截弹与武器系统对应关系'])))
            # 提取WTA数据
            self.vt = (torch.tensor(ast.literal_eval(row['目标威胁度'])))
            pij = re.sub(r'\s+', ',', row['打击概率pij'])  # 将连续的空格替换为逗号
            pij = pij.replace("] [", "], [")  # 替换每个数据对之间的空格
            self.pij = (torch.tensor(ast.literal_eval(pij)))
            fij = re.sub(r'\s+', ',', row['可行性矩阵'])  # 将连续的空格替换为逗号
            fij = fij.replace("] [", "], [")  # 替换每个数据对之间的空格
            self.fij = (torch.tensor(ast.literal_eval(fij)))
            qij = re.sub(r'\s+', ',', row['打击概率qjk'])  # 将连续的空格替换为逗号
            qij = qij.replace("] [", "], [")  # 替换每个数据对之间的空格
            self.qjk = (torch.tensor(ast.literal_eval(qij)))
            self.vb = (torch.tensor(ast.literal_eval(row['基地价值']))).squeeze()
            # 提取打击方案
            self.plan = (torch.tensor(ast.literal_eval(row['打击方案'])))
        num_target = len(self.target_type)
        num_weapon = len(self.plan)
        self.num_target = num_target
        self.num_weapon = num_weapon
        Randint = len(self.vb)

        self.init_WTA_input(num_weapon, num_target, num_samples, seed)
        self.init_blue_input(num_weapon)
        self.init_red_input(num_target)

        self.num_nodes = num_weapon * num_target
        self.size = num_samples
        # 恢复成矩阵形式方便计算
        self.Pij = self.pij.reshape(num_samples, num_weapon, num_target)
        self.Threat = self.vt
        self.target_pij2 = self.qjk.squeeze()
        self.Qjk = self.qjk


    def __len__(self):
        return self.size

    def init_WTA_input(self, num_weapon, num_target, num_samples, seed):
        # WTA信息处理
        Randint = 3  # 基地的数量
        positions = []
        # 遍历 self.qjk 中的每个张量
        for tensor in self.qjk:
            # 使用 nonzero() 找到不为零的元素的位置
            nonzero_indices = torch.nonzero(tensor)
            # 选择每行的第一个非零元素的索引
            if nonzero_indices.size(0) > 0:
                # 获取第一个非零元素的位置
                first_nonzero_pos = nonzero_indices[0]
                positions.append(first_nonzero_pos)
        # 将位置合并为一个 (1, 2) 形状的张量
        self.T_to_A = torch.stack(positions).view(1, -1)
        # 基地的价值的标准化
        self.norm_value = self.vb / 100
        # 目标对应基地的频率， 基地的价值占比
        self.f, self.base = trans_norm(self.T_to_A, Randint, self.norm_value)
        ##################
        self.f1 = self.f.reshape(num_samples, 1, num_target).repeat(1, 1, num_weapon)
        self.base1 = self.base.reshape(num_samples, 1, num_target).repeat(1, 1, num_weapon)
        #####################
        torch.manual_seed(seed + 10)
        # 提取每一行的不为零的元素
        target_pij = []
        for row in self.qjk:
            # 获取该行中不为零的元素
            non_zero_elements = row[row != 0]
            target_pij.append(non_zero_elements[0])
        # 将提取的非零元素合并为一个张量
        self.target_pij = torch.stack(target_pij).unsqueeze(0).unsqueeze(0)
        # 目标打击基地的概率
        self.target_pij1 = self.target_pij.repeat(1, 1, num_weapon)
        torch.manual_seed(seed + 20)
        reshaped_pij = self.pij.view(-1)  # 这将把 (32, 2) 张量转换为一个 64 元素的一维张量
        self.weapon_pij = reshaped_pij.unsqueeze(0).unsqueeze(0)
        reshaped_fij = self.fij.view(-1)  # 这将把 (32, 2) 张量转换为一个 64 元素的一维张量
        self.weapon_fij = reshaped_fij.unsqueeze(0).unsqueeze(0)
        # 武器打击目标的概率
        torch.manual_seed(seed + 30)
        # 目标的威胁值
        repeated_vt = self.vt.repeat(num_weapon)
        self.target_threat1 = repeated_vt.unsqueeze(0).unsqueeze(0)
        # WTA相关信息输入
        self.dataset1 = torch.cat((self.f1, self.base1, self.target_pij1, self.weapon_pij, self.weapon_fij,
                                   self.target_threat1, self.weapon_pij), dim=1)

    def init_blue_input(self, num_weapon):
        # 蓝方信息处理
        # 目标类型
        repeated_target_type = self.target_type.repeat(num_weapon)
        self.target_type1 = repeated_target_type.unsqueeze(0).unsqueeze(0)
        # 目标速度
        repeated_target_speed = self.target_speed.repeat(num_weapon)
        self.target_speed1 = repeated_target_speed.unsqueeze(0).unsqueeze(0)
        # 目标方位角
        repeated_target_azimuth = self.target_azimuth.repeat(num_weapon)
        self.target_azimuth1 = repeated_target_azimuth.unsqueeze(0).unsqueeze(0)
        # 作战范围
        repeated_target_range = self.target_range.repeat(num_weapon)
        self.target_range1 = repeated_target_range.unsqueeze(0).unsqueeze(0)
        # 目标经度
        longitude = self.target_coordinates[:, 0]  # 获取每行的第一个元素（经度）
        repeated_target_longitude = longitude.repeat(num_weapon)
        self.target_longitude1 = repeated_target_longitude.unsqueeze(0).unsqueeze(0)
        # 目标纬度
        latitude = self.target_coordinates[:, 1]  # 获取每行的第一个元素（经度）
        repeated_target_latitude = latitude.repeat(num_weapon)
        self.target_latitude1 = repeated_target_latitude.unsqueeze(0).unsqueeze(0)
        # 蓝方信息输入
        self.dataset2 = torch.cat(
            (self.target_type1, self.target_speed1, self.target_azimuth1, self.target_range1, self.target_longitude1,
             self.target_latitude1), dim=1)

    def init_red_input(self, num_target):
        self.weapon_type_wta = self.weapon_type[self.weapon_relation]
        self.weapon_coordinates_wta = self.weapon_coordinates[self.weapon_relation]
        self.weapon_damage_wta = self.weapon_damage[self.weapon_relation]
        self.weapon_range_wta = self.weapon_range[self.weapon_relation]
        self.weapon_ammunition_wta = self.weapon_ammunition[self.weapon_relation]
        self.weapon_weight_wta = self.weapon_weight[self.weapon_relation]
        # 红方信息处理
        # 武器类型
        repeated_weapon_type = self.weapon_type_wta.repeat_interleave(num_target)
        self.weapon_type1 = repeated_weapon_type.unsqueeze(0).unsqueeze(0)
        # 武器经度
        longitude = self.weapon_coordinates_wta[:, 0]  # 获取每行的第一个元素（经度）
        repeated_weapon_longitude = longitude.repeat_interleave(num_target)
        self.weapon_longitude1 = repeated_weapon_longitude.unsqueeze(0).unsqueeze(0)
        # 武器纬度
        latitude = self.weapon_coordinates_wta[:, 1]  # 获取每行的第一个元素（经度）
        repeated_weapon_latitude = latitude.repeat_interleave(num_target)
        self.weapon_latitude1 = repeated_weapon_latitude.unsqueeze(0).unsqueeze(0)
        # 武器毁伤程度
        repeated_weapon_damage = self.weapon_damage_wta.repeat_interleave(num_target)
        self.weapon_damage1 = repeated_weapon_damage.unsqueeze(0).unsqueeze(0)
        # 武器打击范围
        repeated_weapon_range = self.weapon_range_wta.repeat_interleave(num_target)
        self.weapon_range1 = repeated_weapon_range.unsqueeze(0).unsqueeze(0)
        # 弹药类型
        repeated_weapon_ammunition = self.weapon_ammunition_wta.repeat_interleave(num_target)
        self.weapon_ammunition1 = repeated_weapon_ammunition.unsqueeze(0).unsqueeze(0)
        # 武器部火药重量
        repeated_weapon_weight = self.weapon_weight_wta.repeat_interleave(num_target)
        self.weapon_weight1 = repeated_weapon_weight.unsqueeze(0).unsqueeze(0)
        # 红方信息输入
        self.dataset3 = torch.cat(
            (self.weapon_type1, self.weapon_longitude1, self.weapon_latitude1, self.weapon_damage1, self.weapon_range1,
             self.weapon_ammunition1, self.weapon_weight1), dim=1)



    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        # 直接通过索引返回数据
        return (self.dataset1[idx], [], self.dataset2[idx], self.dataset3[idx],
                self.Pij[idx], self.Threat, self.Qjk, self.norm_value, torch.tensor(self.num_weapon).to(device),
        torch.tensor(self.num_target).to(device), self.plan)


def pro_wta_reward(static, tour_indices):
    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1)
    idx = idx.repeat(1, 4, 1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)  # [256,20,4]
    v = tour[:, :, 0]  # 基地价值占比
    s = tour[:, :, 1]  # 目标打击基地概率
    t = tour[:, :, 2]  # 目标威胁值
    p = tour[:, :, 3]  # 武器打击目标概率
    average = t * p + v * (1 - s + s * p)
    # Euclidean distance between each consecutive point
    return torch.div(1, average.sum(1)).detach()


def trans_norm(tensor_input, num_base, base_value):
    """
    input:
    tensor_input: n数量个目标打击基地的选择
    num_base: 基地数量
    base_value:基地价值的标准化
    output:
    random_norm:标准化基地选择，利用出现频率代替
    random_norm_base:目标打击基地价值的标准化

    """
    B = []
    C = []
    # num_sample
    row = int(tensor_input.size(0))
    # target
    column = int(tensor_input.size(1))
    random_norm = torch.zeros((row, column))
    random_norm_base = torch.zeros((row, column))
    for i in range(row):
        for j in range(num_base):
            count = torch.sum(torch.eq(tensor_input[i, :], j).int())
            B.append(int(count))
        C.append(B)
        B = []
    for i in range(row):
        for j in range(column):
            index = int(tensor_input[i][j])
            random_norm[i][j] = 1 / C[i][index]
            random_norm_base[i][j] = base_value[index]
    return random_norm, random_norm_base


def compute_reward(Pij, Threat, Qjk, V_a, plan, excu_time, weapon_cool):
    Qjk_start = 1 * Qjk
    Sur_Pij = -1 * Pij + 1
    Tar_to_Asset = np.nonzero(Qjk)[1]
    Pij_start = np.ones_like(Pij)
    sum_Threat = sum(Threat)
    if Sur_Pij.ndim == 3 and Sur_Pij.shape[0] == 1:
        Sur_Pij = Sur_Pij.squeeze(0)
    # 计算消除的威胁值
    for i, temp1 in enumerate(plan):
        if temp1 >= 0:
            Pij_start[i][temp1] = Sur_Pij[i][temp1]
        if 0 <= temp1 < len(Tar_to_Asset):
            temp = Tar_to_Asset[temp1]
            Qjk_start[temp1][temp] = Sur_Pij[i][temp1] * Qjk_start[temp1][temp]
    for target_index, item in enumerate(excu_time):
        if item == 1:
            Threat[target_index] = 0
    Pij_end = np.prod(Pij_start, axis=0)
    T_sum = ((-1 * Pij_end + 1) * Threat).sum()
    Qjk_new = -1 * Qjk_start + 1
    Q_end = np.prod(Qjk_new, axis=0)
    V_sum = (Q_end * V_a).sum()
    # 新加的前瞻性目标函数
    f3 = 0
    pij_sur = np.ones((1, len(Threat)))
    fij_next = np.ones((len(plan), len(Threat)))
    for weapon, target in enumerate(plan):
        if target != -1:
            pij_sur[0][target] *= (1 - Pij[weapon][target])
            if weapon_cool[weapon] == 1:
                fij_next[weapon, :] = 0
            if excu_time[target] == 1:
                pij_sur[0][target] = 1
    for target in range(len(Threat)):
        if excu_time[target] == 0:
            fij_next[:, target] = 0
    a1 = 0
    a2 = 0
    for target in range(len(Threat)):
        a1 += Threat[target] * pij_sur[0][target]
        a3 = 0
        m = 0
        for weapon in range(len(plan)):
            a3 += Pij[weapon][target] * fij_next[weapon][target]
            m += fij_next[weapon][target]
        if m != 0:
            a2 += Threat[target] * pij_sur[0][target] * (a3 / m)
        else:
            a2 = 0
    if a1 != 0:
        f3 = (1 / a1) * a2
    else:
        f3 = 1

    return T_sum / sum_Threat + V_sum / sum(V_a)
    # return T_sum / sum_Threat + V_sum / sum(V_a) + f3


def compute_batch(B_pij, B_threat, B_qjk, B_va, B_plan, B_excu_time, B_weapon_cool):
    bsize = B_pij.size(0)
    Reward = []
    for b in range(bsize):
        pij = B_pij[b].to('cpu').numpy()
        threat = B_threat[b].to('cpu').numpy()[0]
        qjk = B_qjk[b].to('cpu').numpy()
        va = B_va.to('cpu').numpy()[0]
        plan = B_plan[b].to('cpu').numpy()

        excu_time = B_excu_time[b].to('cpu').numpy()[0]
        weapon_cool = B_weapon_cool[b].to('cpu').numpy()[0]

        reward = compute_reward(pij, threat, qjk, va, plan, excu_time, weapon_cool)
        Reward.append(1 / reward)
    return torch.tensor(Reward).to(device)


def trans_to_plan(index, num_weapon, num_target):
    batch_size = index.size(0)

    # 新建一个固定形状的 tensor：全部初始化为 -1，代表未分配
    index_sort = torch.full((batch_size, num_weapon), -1, dtype=torch.long, device=index.device)

    for row, v in enumerate(index):
        temp_sort, _ = torch.sort(index[row])

        # 计算武器编号和目标编号
        weapon_index = torch.div(temp_sort, num_target, rounding_mode='floor').tolist()
        target_index = torch.remainder(temp_sort, num_target).tolist()

        # 初始化分配方案
        plan = [-1] * num_weapon
        for k, w in zip(weapon_index, target_index):
            if 0 <= k < num_weapon:
                plan[k] = w  # 第 k 个武器打第 w 个目标

        # 替换当前行
        index_sort[row] = torch.tensor(plan, device=index.device)

    return index_sort


def distance(human_plan, agent_plan):
    # 去掉额外的维度，确保输入是1D张量
    human_plan = human_plan.squeeze(0).to(device).to(torch.float32)
    agent_plan = agent_plan.squeeze(0).to(device).to(torch.float32)

    # 计算欧几里得距离
    dist = torch.norm(human_plan - agent_plan, p=2)  # p=2表示欧几里得距离

    return dist

def entropy_regularization_loss(logp, lambda_ = 0.5):
    # 计算P(a|s)，通过对logP(a|s)取指数
    p = torch.exp(logp)

    # 计算熵正则化损失
    entropy_loss = -lambda_ * torch.sum(p * logp)

    return entropy_loss


def cosine_similarity_percentage(human_plan, agent_plan):
    # 去掉额外的维度，确保输入是1D张量
    human_plan = human_plan.squeeze(0).to(device).to(torch.float32)
    agent_plan = agent_plan.squeeze(0).to(device).to(torch.float32)

    # 计算余弦相似度
    dot_product = torch.dot(human_plan, agent_plan)
    human_norm = torch.norm(human_plan)
    agent_norm = torch.norm(agent_plan)

    cosine_sim = dot_product / (human_norm * agent_norm)

    # 将相似度归一化为百分比（0到100之间）
    similarity_percentage = (cosine_sim + 1) / 2 * 100  # 余弦相似度[-1, 1] -> [0, 100]

    return similarity_percentage