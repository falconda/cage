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
# 1
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = '蓝方数据库.xlsx'  # 使用相对路径
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
    # mask.scatter_(1, weapon_mask, 0)
    mask.scatter_(1, target_mask, 0)
    return mask


def parse_tensor_from_string(s):
    if pd.isna(s):  # 空值处理
        return torch.tensor([])

    s = str(s).strip().replace('\n', ' ').replace('\t', ' ')

    # 如果没有逗号，说明是空格分隔的，就替换空格
    if ',' not in s:
        s = re.sub(r'\s+', ',', s)

    s = s.replace("],[", "],[").replace("] [", "],[")  # 修复嵌套数组分隔问题

    try:
        return torch.tensor(ast.literal_eval(s))
    except Exception as e:
        print("解析错误的字符串：", s)
        raise e


class ProWTADataset(Dataset):

    def __init__(self, num_samples=1):
        super(ProWTADataset, self).__init__()

        seed = np.random.randint(123456789)
        global line
        # 这里手动填一下最大行
        line = random.randint(0, 170)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        # 按行遍历前num_samples行
        for index, row in data.iloc[line:num_samples+line].iterrows():
            self.blue_type = parse_tensor_from_string(row['蓝方类型'])
            self.blue_coordinates = parse_tensor_from_string(row['蓝方经纬度'])
            self.blue_speed = parse_tensor_from_string(row['蓝方速度'])
            self.blue_azimuth = parse_tensor_from_string(row['蓝方方位角'])
            self.blue_range = parse_tensor_from_string(row['作战范围'])

            self.red_type = parse_tensor_from_string(row['红方武器类型'])
            self.red_coordinates = parse_tensor_from_string(row['红方武器经纬度'])
            self.red_damage = parse_tensor_from_string(row['武器毁伤程度'])
            self.red_range = parse_tensor_from_string(row['打击范围'])
            self.red_ammunition = parse_tensor_from_string(row['弹药类型'])
            self.red_weight = parse_tensor_from_string(row['武器部火药重量'])

            self.vt = parse_tensor_from_string(row['红方威胁度'])
            self.pij = parse_tensor_from_string(row['打击概率pij'])
            self.fij = parse_tensor_from_string(row['可行性矩阵'])
            self.plan = process_plan(parse_tensor_from_string(row['打击方案']))
            self.qjk = parse_tensor_from_string(row['损伤概率qjk'])

            # print(self.red_coordinates.shape)  # 应该是 torch.Size([32, 2])
            # print(self.red_coordinates[0])  # 看某一行是否真的含有 \n
        num_red = len(self.red_type)
        num_blue = len(self.blue_type)
        self.num_target = num_red
        self.num_weapon = num_blue
        # Randint = len(self.vb)

        self.init_WTA_input(num_blue, num_red, num_samples, seed)
        self.init_blue_input(num_red)
        self.init_red_input(num_blue)

        self.num_nodes = num_blue * num_red
        self.size = num_samples
        # 恢复成矩阵形式方便计算
        self.Pij = self.pij.reshape(num_samples, num_blue, num_red)
        self.Threat = self.vt
        self.target_pij2 = self.qjk.squeeze()
        self.Qjk = self.qjk


    def __len__(self):
        return self.size

    def init_WTA_input(self, num_weapon, num_target, num_samples, seed):
        # 目标的威胁值
        repeated_vt = self.vt.repeat(num_weapon)
        repeated_vt_arr = repeated_vt.view(-1)
        self.target_threat1 = repeated_vt_arr.unsqueeze(0).unsqueeze(0)
        pij1_arr = self.pij.view(-1)
        self.pij1 = pij1_arr.unsqueeze(0).unsqueeze(0)
        fij1_arr = self.fij.view(-1)
        self.fij1 = fij1_arr.unsqueeze(0).unsqueeze(0)
        qjk1_arr = self.qjk.view(-1)
        self.qjk1 = qjk1_arr.unsqueeze(0).unsqueeze(0)

        # WTA相关信息输入
        self.dataset1 = torch.cat((self.target_threat1, self.pij1, self.fij1, self.qjk1), dim=1)

    def init_blue_input(self, num_red):
        # 蓝方信息处理
        # 目标类型
        repeated_blue_type = self.blue_type.unsqueeze(1).repeat(1,num_red)
        self.blue_type1 = repeated_blue_type.view(-1).unsqueeze(0).unsqueeze(0)
        # 目标速度
        repeated_blue_speed = self.blue_speed.unsqueeze(1).repeat(1,num_red)
        self.blue_speed1 = repeated_blue_speed.view(-1).unsqueeze(0).unsqueeze(0)
        # 目标方位角
        repeated_blue_azimuth = self.blue_azimuth.unsqueeze(1).repeat(1,num_red)
        self.blue_azimuth1 = repeated_blue_azimuth.view(-1).unsqueeze(0).unsqueeze(0)
        # 作战范围
        repeated_blue_range = self.blue_range.unsqueeze(1).repeat(1,num_red)
        self.blue_range1 = repeated_blue_range.view(-1).unsqueeze(0).unsqueeze(0)
        # 目标经度
        longitude = self.blue_coordinates[:, 0]  # 获取每行的第一个元素（经度）
        repeated_blue_longitude = longitude.unsqueeze(1).repeat(1,num_red)
        self.blue_longitude1 = repeated_blue_longitude.view(-1).unsqueeze(0).unsqueeze(0)
        # 目标纬度
        latitude = self.blue_coordinates[:, 1]  # 获取每行的第一个元素（经度）
        repeated_blue_latitude = latitude.unsqueeze(1).repeat(1,num_red)
        self.blue_latitude1 = repeated_blue_latitude.view(-1).unsqueeze(0).unsqueeze(0)
        # 蓝方信息输入
        self.dataset2 = torch.cat(
            (self.blue_type1, self.blue_speed1, self.blue_azimuth1, self.blue_range1, self.blue_longitude1,
             self.blue_latitude1), dim=1)

    def init_red_input(self, num_blue):
        self.red_type_wta = self.red_type
        self.red_coordinates_wta = self.red_coordinates
        self.red_damage_wta = self.red_damage
        self.red_range_wta = self.red_range
        self.red_ammunition_wta = self.red_ammunition
        self.red_weight_wta = self.red_weight
        # 红方信息处理
        # 武器类型
        repeated_red_type = self.red_type_wta.repeat(num_blue,1)
        self.red_type1 = repeated_red_type.view(-1).unsqueeze(0).unsqueeze(0)
        # 武器经度
        longitude = self.red_coordinates_wta[:, 0]  # 获取每行的第一个元素（经度）
        repeated_red_longitude = longitude.repeat(num_blue,1)
        self.red_longitude1 = repeated_red_longitude.view(-1).unsqueeze(0).unsqueeze(0)
        # 武器纬度
        latitude = self.red_coordinates_wta[:, 1]  # 获取每行的第一个元素（经度）
        repeated_red_latitude = latitude.repeat(num_blue,1)
        self.red_latitude1 = repeated_red_latitude.view(-1).unsqueeze(0).unsqueeze(0)
        # 武器毁伤程度
        repeated_red_damage = self.red_damage_wta.repeat(num_blue,1)
        self.red_damage1 = repeated_red_damage.view(-1).unsqueeze(0).unsqueeze(0)
        # 武器打击范围
        repeated_red_range = self.red_range_wta.repeat(num_blue,1)
        self.red_range1 = repeated_red_range.view(-1).unsqueeze(0).unsqueeze(0)
        # 弹药类型
        repeated_red_ammunition = self.red_ammunition_wta.repeat(num_blue,1)
        self.red_ammunition1 = repeated_red_ammunition.view(-1).unsqueeze(0).unsqueeze(0)
        # 武器部火药重量
        repeated_red_weight = self.red_weight_wta.repeat(num_blue,1)
        self.red_weight1 = repeated_red_weight.view(-1).unsqueeze(0).unsqueeze(0)
        # 红方信息输入
        self.dataset3 = torch.cat(
            (self.red_type1, self.red_longitude1, self.red_latitude1, self.red_damage1, self.red_range1,
             self.red_ammunition1, self.red_weight1), dim=1)



    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        # 直接通过索引返回数据
        return (self.dataset1[idx], [], self.dataset2[idx], self.dataset3[idx],
                torch.tensor(self.num_weapon).to(device), torch.tensor(self.num_target).to(device),
                self.Threat, self.Pij,  self.Qjk, self.plan)



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


def process_plan(plan):
    # 将每行的非零元素设置为 1，并记录该位置（索引）在新的一维张量中
    rows, cols = plan.size()

    # 用来存储结果的一维张量
    non_zero_indices = []

    for i in range(rows):
        # 获取当前行
        row = plan[i]

        # 将非零元素设置为 1
        row[row != 0] = 1

        # 将当前行的非零元素所在的列索引添加到 non_zero_indices 中
        non_zero_index = torch.nonzero(row).squeeze().tolist()  # 获取不为0的元素的列索引

        # 如果这一行是空的（即没有非零元素），将其索引设置为 -1
        if not non_zero_index:
            non_zero_indices.append(-1)  # 如果该行全是零，添加 -1
        else:
            non_zero_indices.append(non_zero_index)  # 只取每行中第一个非零元素的列索引

    # 将 non_zero_indices 转化为一维张量
    non_zero_indices_tensor = torch.tensor(non_zero_indices)

    return non_zero_indices_tensor

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


def soft_hamming_loss(human_plan, agent_plan, num_targets):
    """
    human_plan, agent_plan: [B, num_weapons]，其中 -1 表示未分配
    num_targets: 原始目标数 N（不包括 -1），我们会 +1 用于 one-hot
    """
    # 全部加 1，让 -1 → 0，其它 +1，对应目标 1~N
    human_plan_shifted = human_plan + 1
    agent_plan_shifted = agent_plan + 1

    num_classes = int(num_targets + 1)  # 新的一类：未分配变成 class 0

    # one-hot 编码后比较
    human_onehot = F.one_hot(human_plan_shifted, num_classes=num_classes).float().to(device)
    agent_onehot = F.one_hot(agent_plan_shifted, num_classes=num_classes).float().to(device)

    # 计算每个武器预测错误的程度
    diff = torch.abs(agent_onehot - human_onehot).sum(dim=-1).to(device)  # [B, num_weapons]

    # 只对原始 human_plan ≠ -1 的位置进行损失计算
    valid_mask = (human_plan >= 0).float().to(device)
    masked_diff = diff * valid_mask

    # 取平均损失
    loss = masked_diff.sum() / valid_mask.sum().clamp(min=1.0)

    return loss.clone().detach().requires_grad_(True).to(device)


def entropy_regularization_loss(logp, lambda_=0.5):
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
    if len(human_plan) == 1 and len(agent_plan) == 1:
        if human_plan[0] == agent_plan[0] == 0:
            similarity_percentage = torch.tensor(1.0).to(device)
        else:
            similarity_percentage = torch.tensor(0.0).to(device)
    if similarity_percentage.isnan == 1:
        print(f'human_plan=', human_plan)
        print(f'agent_plan=', agent_plan)

    return similarity_percentage


def cosine_similarity_loss(human_plan, agent_plan):
    sim = cosine_similarity_percentage(human_plan, agent_plan)  # 输出百分比
    return ((100.0 - sim) / 100.0).to(device)  # 把相似度变成损失（归一化到 0~1）


def soft_hamming_similarity(human_plan, agent_plan, num_targets):
    """
    计算与软汉明距离相关的相似度，结果在 [0, 1] 范围内，且相似度尽量较大。
    human_plan, agent_plan: [B, num_weapons]，其中 -1 表示未分配
    num_targets: 原始目标数 N（不包括 -1），我们会 +1 用于 one-hot
    """
    # 全部加 1，让 -1 → 0，其它 +1，对应目标 1~N
    human_plan_shifted = human_plan + 1
    agent_plan_shifted = agent_plan + 1

    num_classes = int(num_targets + 1)  # 新的一类：未分配变成 class 0

    # one-hot 编码后比较
    human_onehot = F.one_hot(human_plan_shifted, num_classes=num_classes).float().to(device)
    agent_onehot = F.one_hot(agent_plan_shifted, num_classes=num_classes).float().to(device)

    # 计算每个武器预测错误的程度
    diff = torch.abs(agent_onehot - human_onehot).sum(dim=-1).to(device)  # [B, num_weapons]

    # 只对原始 human_plan ≠ -1 的位置进行损失计算
    valid_mask = (human_plan >= 0).float().to(device)
    masked_diff = diff * valid_mask

    # 计算总的误差
    total_error = masked_diff.sum() / valid_mask.sum().clamp(min=1.0)

    # 通过反转损失来获得相似度，确保结果在 [0, 1] 之间
    similarity = 1.0 / (1.0 + total_error)  # 使用加1平滑避免除零

    # 为了让相似度更大一些，可以应用一种加权策略或平滑处理
    # 平滑处理：增强相似度，避免过小的数值
    similarity = torch.clamp(similarity, min=0.0, max=1.0)  # 确保相似度在 [0, 1] 之间

    return similarity


def compute_similarity(human_plan, agent_plan):
    """
    遍历 human_plan 和 agent_plan，若同一位置的元素值相同，则加1分
    最终返回得分 / 总分，其中总分即为序列的长度
    """
    # 确保两者长度相同
    """
    计算 human_plan 和 agent_plan 中非 -1 且位置相同的元素占比（相似度）

    Args:
        human_plan: tensor，[B, N] 或 [N]
        agent_plan: tensor，[B, N] 或 [N]

    Returns:
        similarity_score: float，两个序列在有效比较区域的匹配比例
    """
    # 去掉 batch 维度并转为 float32（如果有）
    human_plan = human_plan.squeeze(0).to(device).to(torch.float32)
    agent_plan = agent_plan.squeeze(0).to(device).to(torch.float32)

    assert human_plan.shape == agent_plan.shape, "human_plan 和 agent_plan 必须具有相同形状"

    # 找出 human_plan 中非 -1 的位置
    valid_mask = (human_plan != -1)

    # 在这些位置上比较值是否相等
    matches = (human_plan == agent_plan) & valid_mask
    match_count = matches.sum().float()
    total_valid = valid_mask.sum().float().clamp(min=1.0)  # 防止除零

    similarity = match_count / total_valid

    # 不相似度
    return 1-similarity
