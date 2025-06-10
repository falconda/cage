import random
import numpy as np
# 当处理规模不同时，种群要增大
class PSO_GA(object):
    def __init__(self, num_weapon, num_target, Wei, Pij, Fij, Qjk, V_a):
        Pij = Pij * Fij
        self.num_target = num_target  # 目标数
        self.num_weapon = num_weapon  # 武器数
        self.MAX = max(num_target, num_weapon)
        self.iter_max = int(3.5 * self.MAX)  # 迭代数目
        self.num = self.MAX  # 粒子数目
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
        # 初始化所有粒子
        #self.particles = self.random_init(self.num)
        #一个粒子群体的的集合
        self.particles = self.greedy_init(self.num, self.MAX)
        #对应粒子的适应值的集合
        self.fitnesses = self.compute_plans(self.particles)
        # 得到初始化群体的最优解
        init_l = max(self.fitnesses)
        init_index = self.fitnesses.index(init_l)
        init_plan = self.particles[init_index]
        # 记录每个个体的当前最优解
        self.local_best = self.particles
        self.local_best_len = self.fitnesses
        # 记录当前的全局最优解,长度是iteration
        self.global_best = init_plan
        self.global_best_len = init_l
        # 输出解
        self.best_l = self.global_best_len
        self.best_plan = self.global_best
        # 存储每次迭代的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [init_l]
        return

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
            add = self.num_target-self.num_weapon
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
    def compute_plan_all(self, plan):
        Qjk_start = 1 * self.Qjk
        Pij_start = np.ones_like(self.Pij)
        # 计算消除的威胁值
        for i in range(len(plan)):
            temp1 = plan[i]
            Pij_start[i][temp1] = self.Sur_Pij[i][temp1]
            if temp1 < len(self.Tar_to_Asset):
                temp = self.Tar_to_Asset[plan[i]]
                Qjk_start[temp1][temp] = self.Sur_Pij[i][temp1] * Qjk_start[temp1][temp]
        Pij_end = np.prod(Pij_start, axis=0)
        T_sum = ((-1 * Pij_end + 1) * self.Wei[0]).sum()
        Qjk_new = -1 * Qjk_start + 1
        Q_end = np.prod(Qjk_new, axis=0)
        V_sum = (Q_end * self.V_a[0]).sum()
        return T_sum / self.Threat_sum + V_sum / self.V_asset_sum
    # 快速计算一对一的值
    def compute_plan(self, plan):
        T_sum = 0
        for v, k in enumerate(plan):
            T_sum = T_sum + self.DES[v][k]
        Qjk_start = 1 * self.Qjk
        # 计算剩余基地价值
        for i, temp1 in enumerate(plan):
            if temp1 < len(self.Tar_to_Asset):
                temp = self.Tar_to_Asset[plan[i]]
                Qjk_start[temp1][temp] = self.Sur_Pij[i][temp1] * Qjk_start[temp1][temp]
        Qjk_new = -1 * Qjk_start + 1
        Q_end = np.prod(Qjk_new, axis=0)
        V_sum = (Q_end * self.V_a[0]).sum()
        return T_sum / self.Threat_sum + V_sum / self.V_asset_sum

    # 计算一个群体的长度
    def compute_plans(self, plans):
        result = []
        for one in plans:
            fitness = self.compute_plan(one)
            result.append(fitness)
        return result

    # 评估当前的群体
    def eval_particles(self):
        max_fitness = max(self.fitnesses)
        max_index = self.fitnesses.index(max_fitness)
        cur_plan = self.particles[max_index]
        # 更新当前的全局最优
        if max_fitness > self.global_best_len:
            self.global_best_len = max_fitness
            self.global_best = cur_plan
        # 更新当前的个体最优
        for i, l in enumerate(self.fitnesses):
            if l > self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particles[i]

    # 粒子交叉
    def cross(self, cur, best):
        one = cur.copy()
        fitness = [t for t in range(self.MAX)]
        if len(fitness) > 1:
            t = random.sample(fitness, 2)
        else:
            t = [0]
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)
        temp1 = tmp[:x]
        temp2 = tmp[x:]
        # 另一种
        random.shuffle(tmp)
        tep_re1 = tmp[:x]
        tep_re2 = tmp[x:]
        # 两种交叉方法
        one = temp1 + cross_part + temp2
        l1 = self.compute_plan(one)
        one2 = tep_re1 + cross_part + tep_re2
        l2 = self.compute_plan(one2)
        if l1 > l2:
            return one, l1
        else:
            return one2, l2
    # 粒子变异
    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.MAX)]
        if len(l) > 1:
            t = random.sample(l, 2)
        else:
            t = [0]
        x, y = min(t), max(t)
        one[x], one[y] = one[y], one[x]
        l2 = self.compute_plan(one)
        return one, l2
    # 迭代操作
    def pso_ga(self):
        for cnt in range(1, self.iter_max):
            # 更新粒子群
            for i, one in enumerate(self.particles):
                tmp_l = self.fitnesses[i]
                # 与当前个体局部最优解进行交叉
                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l > tmp_l:
                    one = new_one
                    tmp_l = new_l
                # 与当前全局最优解进行交叉
                new_one, new_l = self.cross(one, self.global_best)
                if new_l > tmp_l:
                    one = new_one
                    tmp_l = new_l
                # 变异
                new_one, new_l = self.mutate(one)
                if new_l > tmp_l:
                    one = new_one
                # 更新该粒子
                self.particles[i] = one
                self.fitnesses[i] = tmp_l
            # 评估粒子群，更新个体局部最优和个体当前全局最优
            self.eval_particles()
            # 更新输出解
            if self.global_best_len > self.best_l:
                self.best_l = self.global_best_len
                self.best_plan = self.global_best
           # print(self.best_l)
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
        return self.best_l, self.best_plan

    def run(self):
        best_fitness, best_plan = self.pso_ga()
        if self.num_target > self.num_weapon:
            best_plan = best_plan[:self.num_weapon]
        elif self.num_target < self.num_weapon:
            for w, t in enumerate(best_plan):
                if t > self.num_target-1:
                    best_plan[w] = -1
        # print(best_fitness)
        return best_plan

