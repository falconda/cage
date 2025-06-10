import random
import numpy as np
import copy
import csv

class LAMGC_WTA(object):
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
        # 算法参数设定
        self.rng = np.random.default_rng(123)  # 用于产生随机数
        self.num_shuffle = 5*num_weapon
        self.alpha = 0.5
        # 信息保存用
        self.pointer1 = False   # 打印两个指标
        self.pointer2 = False    # 打印最后方案的重复的元素

    # 分析收敛性,计算适应度
    def cal_iter(self, single_plan):
        single_plan = single_plan.astype(int)
        asset_value = copy.deepcopy(self.asset_value)
        value = np.zeros((1, self.num_weapon))
        for m in range(len(single_plan)):
            # m:武器编号  single_plan[m]：目标编号
            if single_plan[m] != -1:
                # 最大化剩余基地价值
                # 拦截概率转换为突防概率
                p1 = (1 - self.hit_probability[m][single_plan[m]])
                # 突防概率*损伤概率
                p2 = p1 * self.damage_probability_sum[0][single_plan[m]]
                # 对应打击基地确定：0/1/2
                target_base = self.Tar_to_Asset[single_plan[m]]
                # 对应造成基地损失，得到剩余基地价值
                asset_value[0][target_base] = asset_value[0][target_base] * (1 - p2)

                # 最大化消灭武器目标威胁
                value1 = self.hit_probability[m][single_plan[m]]
                value2 = self.threat_value[0][single_plan[m]]
                value[0][m] = value1 * value2

        a = np.array(asset_value).flatten().tolist()    # 转为只有一个中括号的
        value_base_remain = sum(a)
        value_threat_sum = np.sum(value)
        per1 = value_base_remain / self.asset_value_sum

        per2 = value_threat_sum / self.threat_value_sum
        return per1, per2

    # 单个个体适应度计算
    def cal_single_fitness(self, single_plan):
        p1, p2 = self.cal_iter(single_plan)
        loss = self.alpha * (p1 / self.asset_value_sum) + (1-self.alpha) * (p2 / self.threat_value_sum)
        return loss

    def find_duplicates(self, nums):
        seen = set()
        duplicates = set()
        for num in nums:
            if num in seen:
                duplicates.add(num)
            else:
                seen.add(num)
        return list(duplicates)


    def run(self):
        t = self.threat_value
        q = self.damage_probability_sum
        t_matrix = np.repeat(t, self.num_weapon, axis=0)
        q_matrix = np.repeat(q, self.num_weapon, axis=0)
        p_matrix = self.hit_probability
        d_matrix = t_matrix * q_matrix * p_matrix
        dab_matrix = np.zeros((self.num_weapon, self.num_target))
        e_matrix = np.zeros((self.num_weapon, self.num_target))
        plan = np.full((self.num_shuffle + 1, self.num_weapon), -1)
        fit = np.zeros(self.num_shuffle)
        less = min(self.num_weapon, self.num_target)

        for n in range(self.num_shuffle):
            if n < self.num_shuffle-1:
                u_arr = self.rng.integers(0, 2, size=(self.num_weapon, 1))
            else:
                u_arr = np.full((self.num_weapon, 1), 1)
            # d1_matrix还要一个sign函数,分开进行操作
            d1_matrix = copy.deepcopy(d_matrix * np.sign(u_arr))
            for i in range(less):
                max_index1 = np.argmax(d1_matrix)
                max_row1, max_col1 = np.unravel_index(max_index1, d1_matrix.shape)
                # 计算dab
                dab_matrix = copy.deepcopy(d1_matrix)
                dab_matrix[:, max_col1] = d1_matrix[:, max_col1] * (1 - p_matrix[max_row1, max_col1])
                dab_matrix[max_row1, :] = dab_matrix[max_row1, :] * np.sign(u_arr[max_row1][0] - 1)
                dab_matrix[max_row1, max_col1] = d1_matrix[max_row1, max_col1] * (
                            1 - p_matrix[max_row1, max_col1]) * np.sign(u_arr[max_row1][0] - 1)
                # 前瞻收益和eij
                e_matrix[max_row1, :] = d1_matrix[max_row1, :] + dab_matrix[max_row1, :]
                index = np.argmax(e_matrix[max_row1, :])
                # 确定方案
                plan[n, max_row1] = index
                # 对边界收益矩阵进行调整
                d1_matrix[max_row1, :] = 0
                # 一个武器只能选取一个目标，选取后不能再选取
                d1_matrix[:, max_col1] = 0
                u_arr[max_row1][0] = u_arr[max_row1][0] - 1
            fit[n] = self.cal_single_fitness(plan[n, :])

        fit_index = np.argmax(fit)
        best_plan = plan[fit_index, :]
        # 迭代性能计算
        p1, p2 = self.cal_iter(best_plan)
        # print('迭代方案为{}'.format(plan_iter))
        if self.pointer1:
            print('剩余基地价值比例为：{}'.format(p1))
            print('预计打击目标比例为：{}'.format(p2))

        if self.pointer2:
            print(self.find_duplicates(best_plan))

        # print("输出方案为{}".format(best_plan.astype('int32').tolist()))
        return best_plan.astype('int32').tolist()