import copy
import random
import itertools
import sys
sys.path.append('D:\\Workplace\\moziai\\mozi_ai_sdk\\FKFD\\WTA')
from Heuristic_WTA import Heuristic_WTA
import numpy as np

class DF_WTA(object):
    def __init__(self, num_weapon, num_target, Wei, Pij, Fij, Qjk, V_a):
        self.num_weapon = num_weapon
        self.num_target = num_target
        self.threat_value = Wei  # [1, num_target]
        self.hit_probability = Pij  # [num_weapon, num_target]
        self.feasibility = Fij  # [num_weapon, num_target]
        self.damage_probability = Qjk  # [num_target, num_base],num_base=3，并且每一行只有一个元素不为0
        self.asset_value = V_a  # [1, num_base]， num_base=3
        # 目标打击基地的列表
        self.Tar_to_Asset = np.nonzero(self.damage_probability)[1]
        self.damage_probability_sum = np.zeros((1, self.num_target))
        for i in range(self.num_target):
            self.damage_probability_sum[0][i] = sum(self.damage_probability[i, :])
        self.asset_value_sum = sum(V_a[0])
        self.threat_value_sum = sum(Wei[0])
        # 目标函数加权
        self.alpha = 0.5
        # 最大迭代次数


    # 穷举 选出所有方案中适应度函数最大的一个

    def run(self):
        num_weapon = self.num_weapon
        num_target = self.num_target

        # 创建一个 0 到 num_weapon-1 的数组
        arr = [i if i < num_target else -1 for i in range(num_weapon)]

        # 使用 itertools.permutations 逐个生成排列，避免内存问题
        max_fitness = -np.inf  # 初始值设置为负无穷大
        best_plan = None

        # 逐个生成排列，计算适应度并比较
        for perm in itertools.permutations(arr):
            plan = np.array(perm).tolist()
            compute_fitness = Heuristic_WTA(
                self.num_weapon, self.num_target, self.threat_value, self.hit_probability, self.feasibility, self.damage_probability, self.asset_value)
            fit = round(compute_fitness.compute_fitness(plan, a=0.8, b=0.2), 4)

            if fit > max_fitness:
                max_fitness = fit
                best_plan = plan
                print(max_fitness)

        # 返回最佳计划并转换为列表
        print(max_fitness)
        return list(best_plan)
