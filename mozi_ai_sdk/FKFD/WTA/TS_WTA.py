import random
import numpy as np
import math


class TS_WTA(object):
    def __init__(self, num_weapon, num_target, Wei, Pij, Fij, Qjk, V_a):
        Pij = Pij * Fij
        self.num_target = num_target  # 目标数
        self.num_weapon = num_weapon  # 武器数
        self.MAX = max(num_target, num_weapon)
        self.taboo_size = int(self.MAX / 5)
        self.iteration = 1500  # 4
        self.num = 500
        self.wei = Wei
        self.pij = Pij
        self.qjk = Qjk
        self.V_a = V_a
        self.Threat_sum = sum(Wei[0])
        self.V_asset_sum = sum(V_a[0])
        self.correct(self.wei, self.pij, self.qjk)
        self.Sur_Pij = -1 * self.Pij + 1
        self.DES = self.Pij * self.Wei
        # 目标打击基地的列表
        self.Tar_to_Asset = np.nonzero(self.qjk)[1]
        # 禁忌表
        self.taboo = []
        self.plan = self.greedy_init(self.num, self.MAX)
        # self.plan = restriction(self.DES)#self.random_init()
        self.best_plan = self.plan
        self.cur_plan = self.plan
        self.best_fitness = self.compute_planlen(self.plan)

        # 显示初始化后的路径
        # 存储结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [self.best_fitness]

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
            result_one = [0 for _ in range(MAX)]
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
        planlens = self.compute_plans(result)
        sortindex = np.argsort(planlens)
        index = sortindex[0]
        return result[index]

    # 初始化一个随机方案
    def random_init(self):
        tmp = [x for x in range(self.MAX)]
        random.shuffle(tmp)
        return tmp

    def compute_planlen(self, path):
        T_sum = 0
        for v, k in enumerate(path):
            T_sum = T_sum + self.DES[v][k]
        Qjk_start = 1 * self.Qjk
        # 计算剩余基地价值
        for i, temp1 in enumerate(path):
            if temp1 < len(self.Tar_to_Asset):
                temp = self.Tar_to_Asset[temp1]
                Qjk_start[temp1][temp] = self.Sur_Pij[i][temp1] * Qjk_start[temp1][temp]
        Qjk_new = -1 * Qjk_start + 1
        Q_end = np.prod(Qjk_new, axis=0)
        V_sum = (Q_end * self.V_a[0]).sum()
        return 1 * T_sum / self.Threat_sum + 1 * V_sum / self.V_asset_sum

    # 计算一个群体的长度
    def compute_plans(self, plans):
        result = []
        for one in plans:
            fitness = self.compute_planlen(one)
            result.append(fitness)
        return result

    # 产生随机解
    def ts_search(self, x):
        moves = []
        new_plans = []
        while len(new_plans) < self.MAX * 2:
            i = np.random.randint(len(x))
            j = np.random.randint(len(x))
            tmp = x.copy()
            tmp[i], tmp[j] = tmp[j], tmp[i]
            new_plans.append(tmp)
            moves.append([i, j])
        return new_plans, moves

    # 禁忌搜索
    def ts(self):
        for cnt in range(self.iteration):
            new_plans, moves = self.ts_search(self.cur_plan)
            new_fitnesss = self.compute_plans(new_plans)
            sort_index = np.argsort(new_fitnesss)
            sort_index = list(sort_index)
            sort_index = sort_index[::-1]
            max_l = new_fitnesss[sort_index[0]]
            max_plan = new_plans[sort_index[0]]
            max_move = moves[sort_index[0]]

            # 更新当前的最优路径
            if max_l > self.best_fitness:
                self.best_fitness = max_l
                self.best_plan = max_plan
                self.cur_plan = max_plan
                # 更新禁忌表
                if max_move in self.taboo:
                    self.taboo.remove(max_move)

                self.taboo.append(max_move)
            else:
                # 找到不在禁忌表中的操作
                while max_move in self.taboo:
                    sort_index = sort_index[1:]
                    max_plan = new_plans[sort_index[0]]
                    max_move = moves[sort_index[0]]
                self.cur_plan = max_plan
                self.taboo.append(max_move)
            # 禁忌表超长了
            if len(self.taboo) > self.taboo_size:
                self.taboo = self.taboo[1:]
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_fitness)
        #    print(cnt, self.best_fitness)
        # print(self.best_fitness)

    def run(self):
        self.ts()
        if self.num_target > self.num_weapon:
            self.best_plan = self.best_plan[:self.num_weapon]
        elif self.num_target < self.num_weapon:
            for w, t in enumerate(self.best_plan):
                if t > self.num_target - 1:
                    self.best_plan[w] = -1
        # print(self.best_plan)
        # print(self.best_fitness)
        return self.best_plan
