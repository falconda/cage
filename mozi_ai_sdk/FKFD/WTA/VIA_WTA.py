import random
import numpy as np
import copy
import csv

class VIA_WTA(object):
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
        self.iter_max = 10
        self.shuffle = 30
        self.plan = np.full((self.iter_max, self.num_weapon), -1)
        self.plan_one = np.full((self.iter_max, self.num_weapon), -1)

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

    # 基于值迭代的WTA
    # state：对应选取多少个武器
    # action：对应选取目标编号
    # state value：0，有状态转移，取值为剩余基地比例
    # actio value：state value为0，加上reward
    # reward：威胁值*打击概率/威胁值

    # 分别以不同编号的武器作为第一行，每一行选取reward值最大的，对reward进行求和，再对所有方法进行适应度排序，选择最大的
    # 计算每个目标的reward
    def run(self):
        num_s = self.num_weapon
        num_a = self.num_target
        num_shuffle = self.shuffle
        MIN = min(num_s, num_a)
        t = self.threat_value
        q = self.damage_probability_sum

        # 威胁值转换成矩阵形式
        matrix_f = np.repeat(t, num_s, axis=0)
        reward_matrix = (matrix_f * self.hit_probability) / matrix_f
        # 计算state_value
        v_value = np.repeat(q, num_s, axis=0)
        # 进行值迭代，其中选择对应action往下的列的reward为0
        # 初始化table, 保存reward和state_value
        table = np.zeros((num_s, num_a))
        strategy_table = np.zeros((num_s, num_a))

        index_arr = np.zeros((num_shuffle, MIN))
        for i in range(self.iter_max):
            plan = np.full((num_shuffle, num_s), -1)
            for n in range(num_shuffle):
                reward = copy.deepcopy(reward_matrix)
                strategy_reward = copy.deepcopy(reward_matrix)
                strategy_table = copy.deepcopy(table)
                # 随机洗牌，打乱选取顺序
                arr = [a for a in range(num_s)]
                random.shuffle(arr)
                # 考虑武器目标数量不一样的情况：
                # 武器多：选择与目标数一样的进行切片，
                # 武器少：不影响，一样使用MIN
                arr1 = np.zeros(MIN)
                if num_s > num_a:
                    arr1 = random.sample(arr, MIN)
                else:
                    arr1 = arr
                index_arr[n, :] = arr1
                # 每行进行操作
                for pos, j in enumerate(arr1):
                    table[j, :] = table[j, :] + self.alpha * reward[j, :]
                    strategy_table[j, :] = strategy_table[j, :] + self.alpha * strategy_reward[j, :]
                    array = strategy_table[j, :]
                    index = np.argmax(array)
                    plan[n, j] = index  # 保存到plan中
                    # 选择的位置会使下一行的strategy_table的一列都为0
                    strategy_reward[:, index] = -1
                    strategy_table[:, index] = -1

                # 最后加上state_value，得到q-value
                table = table + (1 - self.alpha) * v_value
                # 计算得到q-value，赋给操作的矩阵，深拷贝，分离
                '''
                # 洗牌后的顺序还原方案-不用还原
                for a, b in enumerate(arr):
                    sort_plan[n, b] = plan[n, a]                
                '''
            # print('迭代方案为{}'.format(plan))

            #  选取所有洗乱中适应度最大的,作为本次迭代的方案
            fitness = np.zeros(num_shuffle)
            for m in range(num_shuffle):
                fitness[m] = self.cal_single_fitness(plan[m, :])
            index_fit = np.argmax(fitness)
            plan_one = plan[index_fit, :]

            # 历史迭代，选择相比于之前更优秀的
            if i == 0:
                plan_iter = plan_one
            else:
                plan_pre = self.plan_one[i - 1, :]
                fit_now = self.cal_single_fitness(plan_one)
                fit_pre = self.cal_single_fitness(plan_pre)
                if fit_now > fit_pre:
                    plan_iter = plan_one
                else:
                    plan_iter = plan_pre
            self.plan_one[i, :] = plan_iter
            # 迭代性能计算
            p1, p2 = self.cal_iter(plan_iter)
            # print('迭代方案为{}'.format(plan_iter))
            # print('剩余基地价值比例为：{}'.format(p1))
            # print('预计打击目标比例为：{}'.format(p2))
            # 保存数据到csv文件
            '''
            with open('data_VIA_0.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([p1, p2])              
                       
            '''
 
        best_plan = self.plan_one[self.iter_max-1, :].astype('int32')
        best_fit = self.cal_single_fitness(best_plan)
        # print(best_fit)
        # print("输出方案为{}".format(best_plan))
        return best_plan.tolist()