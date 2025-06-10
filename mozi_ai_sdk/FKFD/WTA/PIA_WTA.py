import random
import numpy as np
import copy
import csv

class PIA_WTA(object):
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
        self.iter_max = 50
        self.shuffle = 20
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

    # 基于策略迭代的WTA
    # state：对应选取多少个武器
    # action：对应选取目标编号
    # state value：0，取值为剩余基地比例，由最后的剩余基地价值倒推得到
    # action value：state value + reward
    # reward: 目标函数

    # 分别以不同编号的武器作为第一行，每一行选取reward值最大的，对reward进行求和，再对所有方法进行适应度排序，选择最大的
    # 计算每个目标的reward
    def run(self):
        num_s = self.num_weapon
        num_a = self.num_target
        num_shuffle = self.shuffle
        MIN = min(num_s, num_a)
        t = self.threat_value
        q = 1 - self.damage_probability_sum

        # 威胁值转换成矩阵形式
        matrix_f = np.repeat(t, num_s, axis=0)
        matrix_f1 = (matrix_f * self.hit_probability) / matrix_f
        # 计算state_value
        matrix_q = np.repeat(q, num_s, axis=0)
        # 进行值迭代，其中选择对应action往下的列的reward为0
        reward_matrix = self.alpha * matrix_f1 + (1-self.alpha) * matrix_q
        # 初始化table, 保存reward
        table = np.zeros((num_shuffle, MIN, num_a))

        # 初始化选取的行/列
        index_arr = np.zeros((num_shuffle, MIN))
        plan = np.full((num_shuffle, num_s), -1)
        for count in range(num_shuffle):
            arr = [a for a in range(num_s)]
            random.shuffle(arr)
            arr1 = np.zeros(MIN)
            if num_s > num_a:
                arr1 = random.sample(arr, MIN)
            else:
                arr1 = arr
            # 确定选择行数的序号及顺序
            index_arr[count, :] = arr1

            # 对应方案为【0，1，2】顺序排下？还是洗乱
            if num_s >= num_a:
                item_arr = [b for b in range(num_a)]
            # 目标多：选择武器目标书的目标，随机洗乱进行选取
            elif num_s < num_a:
                arr_temp = [b for b in range(num_a)]
                item_arr = random.sample(arr_temp, MIN)
            random.shuffle(item_arr)

            # 对应方案初始化
            for index, value in enumerate(arr1):
                # 确定初始方案，要计算剩余基地价值作为v_value
                plan[count, value] = item_arr[index]
                # 确定初始化table: 排好顺序的reward
                table[count, index, :] = reward_matrix[value, :]

        # 迭代产生不同方案，得到不同vpaik，从而得到不同的qtable
        for i in range(self.iter_max):
            for n in range(num_shuffle):
                reward = copy.deepcopy(table[n])
                reward_st = copy.deepcopy(table[n])
                selected_array = index_arr[n, :].astype(int)    # 选择状态（武器）的顺序和编号
                selected_table = np.zeros((MIN, num_a))
                selected_table_st = np.zeros((MIN, num_a))
                'policy evaluation'
                # 对于每一个随机洗乱的初始方案：先计算最后v_value: 递归计算得到的
                selected_plan = plan[n, :]
                state_value = np.zeros(num_s)
                state_value[-1], _ = self.cal_iter(selected_plan)
                # 递归计算v_value
                for j in range(MIN-1):    # 从倒数第二个状态进行计算
                    si = -1*(j+1)
                    # 收益：上一个动作确定的
                    ri = selected_plan[si-1]
                    r = reward[si-1, ri]
                    state_value[si-1] = state_value[si] - r
                'policy improvement'
                for a, b in enumerate(selected_array):
                    selected_table[a, :] = selected_table[a, :] + reward[a, :] + state_value[a]
                    selected_table_st[a, :] = selected_table_st[a, :] + reward_st[a, :] + state_value[a]
                    # 索引最大值
                    st_index = np.argmax(selected_table_st[a, :])
                    plan[n, b] = st_index
                    # 之后的列不能为当前列，设置reward为0，q_value为-100
                    reward_st[:, st_index] = 0
                    selected_table_st[:, st_index] = -100

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
            with open('data_PIA_0.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([p1, p2])             
            '''

        best_plan = self.plan_one[self.iter_max-1, :].astype('int32').tolist()
        # print("输出方案为{}".format(best_plan))
        return best_plan