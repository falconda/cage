import random
import numpy as np
import copy


# 当处理规模不同时，种群要增大
class Heuristic_WTA(object):
    def __init__(self, num_weapon, num_target, Wei, Pij, Fij, Qjk, V_a):
        Pij = Pij * Fij
        self.num_target = num_target  # 目标数
        self.num_weapon = num_weapon  # 武器数
        self.MAX = max(num_target, num_weapon)
        self.wei = Wei
        self.pij = Pij
        self.qjk = Qjk
        self.V_a = V_a
        self.Threat_sum = sum(Wei[0])
        self.V_asset_sum = sum(V_a[0])
        self.correct(self.wei, self.pij, self.qjk)
        self.Sur_Pij = -1 * self.Pij + 1
        self.DES = self.Pij * self.Wei
        self.Tar_to_Asset = np.nonzero(self.qjk)[1]
        # 存储每次迭代的结果，画出收敛图
        self.iter_x = []
        self.iter_y = []
        self.iter_y2 = []
        self.iter_y3 = []

    def correct(self, Wei, Pij, Qjk):
        if self.num_weapon > self.num_target:
            add = self.num_weapon - self.num_target
            zero = np.tile(0, (self.num_weapon, add))
            self.Pij = np.append(Pij, zero, axis=1)
            zero_wei = np.tile(0, (1, add))
            self.Wei = np.append(Wei, zero_wei, axis=1)
            zero_qjk = np.tile(0, (add, len(self.V_a[0])))
            self.Qjk = np.append(Qjk, zero_qjk, axis=0)
        elif self.num_weapon < self.num_target:
            add = self.num_target - self.num_weapon
            zero = np.tile(0, (add, self.num_target))
            self.Pij = np.append(Pij, zero, axis=0)
            self.Wei = Wei
            self.Qjk = Qjk
        else:
            self.Pij = Pij
            self.Wei = Wei
            self.Qjk = Qjk

    def greedy_init(self, num_total, MAX):
        result = []
        tmp_choose = 0
        for i in range(num_total):
            result_one = [0] * MAX
            weapon_rest = [x for x in range(0, MAX)]
            target_rest = [y for y in range(0, MAX)]
            while len(target_rest):
                tmp_max = -1
                current = random.choice(weapon_rest)
                weapon_rest.remove(current)
                # 找到威胁度最大目标
                for x in target_rest:
                    if self.DES[current][x] > tmp_max:
                        tmp_max = self.DES[current][x]
                        tmp_choose = x
                result_one[current] = tmp_choose
                target_rest.remove(tmp_choose)
            result.append(result_one)
        return result

    # 随机初始化
    def random_init(self, num_total):
        tmp = [x for x in range(self.MAX)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    # 通用适合计算所有分配的方案
    def compute_fitness(self, plan, a=1, b=1):
        Qjk_start = copy.deepcopy(self.Qjk)
        Pij_start = np.ones_like(self.Pij)
        # 计算消除的威胁值
        for w, t in enumerate(plan):
            Pij_start[w][t] = self.Sur_Pij[w][t]
            if t < len(self.Tar_to_Asset):
                temp = self.Tar_to_Asset[plan[w]]
                Qjk_start[t][temp] = self.Sur_Pij[w][t] * Qjk_start[t][temp]
        Pij_end = np.prod(Pij_start, axis=0)
        T_sum = ((-1 * Pij_end + 1) * self.Wei[0]).sum()
        Qjk_new = -1 * Qjk_start + 1
        Q_end = np.prod(Qjk_new, axis=0)
        V_sum = (Q_end * self.V_a[0]).sum()
        return max(1 * T_sum / self.Threat_sum + 0 * V_sum / self.V_asset_sum, 0.01)

    def compute_fitness2(self, plan, a=1, b=1):
        Qjk_start = copy.deepcopy(self.Qjk)
        Pij_start = np.ones_like(self.Pij)
        # 计算消除的威胁值
        for w, t in enumerate(plan):
            Pij_start[w][t] = self.Sur_Pij[w][t]
            if t < len(self.Tar_to_Asset):
                temp = self.Tar_to_Asset[plan[w]]
                Qjk_start[t][temp] = self.Sur_Pij[w][t] * Qjk_start[t][temp]
        Pij_end = np.prod(Pij_start, axis=0)
        T_sum = ((-1 * Pij_end + 1) * self.Wei[0]).sum()
        Qjk_new = -1 * Qjk_start + 1
        Q_end = np.prod(Qjk_new, axis=0)
        V_sum = (Q_end * self.V_a[0]).sum()
        return max(1 * T_sum / self.Threat_sum + 0 * V_sum / self.V_asset_sum, 0.01)

    def compute_fitness3(self, plan, a=1, b=1):
        Qjk_start = copy.deepcopy(self.Qjk)
        Pij_start = np.ones_like(self.Pij)
        # 计算消除的威胁值
        for w, t in enumerate(plan):
            Pij_start[w][t] = self.Sur_Pij[w][t]
            if t < len(self.Tar_to_Asset):
                temp = self.Tar_to_Asset[plan[w]]
                Qjk_start[t][temp] = self.Sur_Pij[w][t] * Qjk_start[t][temp]
        Pij_end = np.prod(Pij_start, axis=0)
        T_sum = ((-1 * Pij_end + 1) * self.Wei[0]).sum()
        Qjk_new = -1 * Qjk_start + 1
        Q_end = np.prod(Qjk_new, axis=0)
        V_sum = (Q_end * self.V_a[0]).sum()
        return max(0 * T_sum / self.Threat_sum + 1 * V_sum / self.V_asset_sum, 0.01)

    # 计算一个群体的长度
    def group_fitness(self, plans):
        result = []
        for one in plans:
            fitness = self.compute_fitness(one)
            result.append(fitness)
        return result
