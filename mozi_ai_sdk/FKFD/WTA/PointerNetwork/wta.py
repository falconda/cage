import numpy as np
import random
import torch
from torch.utils.data import Dataset
import matplotlib
import torch.nn.functional as F

matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def __init__(self, num_weapon=20, num_target=20, num_samples=1000000, seed=None):
        super(ProWTADataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples

        # 每个目标的可被执行时间 -1:都能执行；0：第一阶段能执行；1：第二阶段能执行
        # 每个目标的可被执行时间（随机生成）
        # 生成目标的可执行时间 (num_samples, 1, num_target)
        self.execution_time = torch.randint(-1, 2, (num_samples, 1, num_target), dtype=torch.int)

        # 生成武器的缓冲时间 (num_samples, 1, num_weapon)
        # 以 80% 的概率生成 0，20% 的概率生成 1
        self.weapon_cooldown = (torch.rand(num_samples, 1, num_weapon) > 0.8).int()

        # 基地的数量
        Randint = 3  # random.randint(0, 10)
        # 目标打击的基地序列
        self.T_to_A = torch.randint(0, Randint, (num_samples, num_target))
        # 基地的价值的标准化
        self.norm_value = torch.rand(Randint)  # V_a
        # 目标对应基地的频率， 基地的价值占比
        self.f, self.base = trans_norm(self.T_to_A, Randint, self.norm_value)
        ##################
        self.f1 = self.f.reshape(num_samples, 1, num_target).repeat(1, 1, num_weapon)
        self.base1 = self.base.reshape(num_samples, 1, num_target).repeat(1, 1, num_weapon)
        #####################
        torch.manual_seed(seed + 10)
        # 目标打击基地的概率
        self.target_pij = torch.rand((num_samples, 1, num_target))
        self.target_pij_update = self.target_pij
        self.target_pij1 = self.target_pij.repeat(1, 1, num_weapon)

        ######################
        self.execution_time1 = self.execution_time.repeat(1, 1, num_weapon)
        self.weapon_cooldown1 = self.weapon_cooldown.repeat(1, 1, num_target)
        ######################


        torch.manual_seed(seed + 20)
        # 武器打击目标的概率
        self.weapon_pij = torch.rand((num_samples, 1, num_weapon * num_target))
        torch.manual_seed(seed + 30)
        # 目标的威胁值
        self.target_threat = torch.rand((num_samples, 1, num_target))
        self.target_threat1 = self.target_threat.repeat(1, 1, num_weapon)
        # 针对基地的匹配对
        self.dataset = torch.cat((self.f1, self.base1, self.target_pij1, self.weapon_pij), dim=1)
        # 针对目标的匹配对
        self.dataset1 = torch.cat((self.target_threat1, self.weapon_pij, self.execution_time1, self.weapon_cooldown1), dim=1)
        self.num_nodes = num_weapon * num_target
        self.size = num_samples
        # 恢复成矩阵形式方便计算
        self.Pij = self.weapon_pij.reshape(num_samples, num_weapon, num_target)
        self.Threat = self.target_threat
        self.target_pij2 = self.target_pij.squeeze()
        self.Qjk = torch.zeros((num_samples, num_target, Randint))
        if num_samples == 1:
            for k in range(num_samples):
                for i, j in enumerate(self.T_to_A[k]):
                    self.Qjk[k][i][j] = self.target_pij2[i]
        else:
            for k in range(num_samples):
                for i, j in enumerate(self.T_to_A[k]):
                    self.Qjk[k][i][j] = self.target_pij2[k][i]

    def __len__(self):
        return self.size

    def update(self, num_weapon, num_target, pij, qjk, vt, execution_time, weapon_cooldown, T_to_A):
        pij = pij.to(self.f1.device).view(-1).unsqueeze(0).unsqueeze(0)
        qjk = qjk.to(self.f1.device).unsqueeze(0).unsqueeze(0)
        vt = vt.to(self.f1.device).unsqueeze(0).unsqueeze(0)
        T_to_A = T_to_A.to(self.f1.device).unsqueeze(0)
        execution_time = execution_time.to(self.f1.device).unsqueeze(0).unsqueeze(0)
        weapon_cooldown = weapon_cooldown.to(self.f1.device).unsqueeze(0).unsqueeze(0)

        num_samples = self.num_samples
        seed = np.random.randint(123456789)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.execution_time = execution_time
        self.weapon_cooldown = weapon_cooldown
        Randint = 3  # random.randint(0, 10)
        self.T_to_A = T_to_A
        self.f, self.base = trans_norm(self.T_to_A, Randint, self.norm_value)
        self.f1 = self.f.reshape(num_samples, 1, num_target).repeat(1, 1, num_weapon)
        self.base1 = self.base.reshape(num_samples, 1, num_target).repeat(1, 1, num_weapon)
        torch.manual_seed(seed + 10)
        # 目标打击基地的概率
        self.target_pij = qjk
        self.target_pij1 = self.target_pij.repeat(1, 1, num_weapon)

        ######################
        self.execution_time1 = self.execution_time.repeat(1, 1, num_weapon)
        self.weapon_cooldown1 = self.weapon_cooldown.repeat(1, 1, num_target)
        ######################

        torch.manual_seed(seed + 20)
        # 武器打击目标的概率
        self.weapon_pij = pij
        torch.manual_seed(seed + 30)
        # 目标的威胁值
        self.target_threat = vt
        self.target_threat1 = self.target_threat.repeat(1, 1, num_weapon)
        # 针对基地的匹配对
        self.dataset = torch.cat((self.f1, self.base1, self.target_pij1, self.weapon_pij), dim=1)
        # 针对目标的匹配对
        self.dataset1 = torch.cat((self.target_threat1, self.weapon_pij, self.execution_time1, self.weapon_cooldown1),
                                  dim=1)
        self.num_nodes = num_weapon * num_target
        self.size = num_samples
        # 恢复成矩阵形式方便计算
        self.Pij = self.weapon_pij.reshape(num_samples, num_weapon, num_target)
        self.Threat = self.target_threat
        self.target_pij2 = self.target_pij.squeeze()
        self.Qjk = torch.zeros((num_samples, num_target, Randint))
        if num_samples == 1:
            if self.target_pij2.dim() == 0:
                self.target_pij2 = self.target_pij2.unsqueeze(0)
            for k in range(num_samples):
                for i, j in enumerate(self.T_to_A[k]):
                    self.Qjk[k][i][j] = self.target_pij2[i].item()
        else:
            for k in range(num_samples):
                for i, j in enumerate(self.T_to_A[k]):
                    self.Qjk[k][i][j] = self.target_pij2[k][i]

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        # 直接通过索引返回数据
        return (self.dataset[idx], [], self.dataset1[idx],
                self.Pij[idx], self.Threat[idx], self.Qjk[idx], self.norm_value, self.execution_time[idx],
                self.weapon_cooldown[idx])


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
    for target_index,item in enumerate(excu_time):
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
        if m!=0:
            a2 += Threat[target] * pij_sur[0][target] * (a3 / m)
        else:
            a2 = 0
    if a1!= 0 :
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
