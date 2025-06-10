import random
import numpy as np
import copy
import csv


class HDGWO_WTA(object):
    def __init__(self, num_weapon, num_target, Wei, Pij, Fij, Qjk, V_a):
        self.num_weapon = num_weapon
        self.num_target = num_target
        self.threat_value = Wei         # [1, num_target]
        self.hit_probability = Pij      # [num_weapon, num_target]
        self.feasibility = Fij          # [num_weapon, num_target]
        self.damage_probability = Qjk   # [num_target, num_base],num_base=3，并且每一行只有一个元素不为0
        self.asset_value = V_a          # [1, num_base]， num_base=3
        self.asset_value_sum = sum(V_a[0])
        self.threat_value_sum = sum(Wei[0])
        # 目标打击基地的列表
        self.Tar_to_Asset = np.nonzero(self.damage_probability)[1]
        self.MAX = max(num_weapon, num_target)
        self.damage_probability_sum = np.zeros((1, self.num_target))
        self.f = np.ones((num_weapon, num_target))

        # 构造Expect
        expect_temp1 = (1 - self.hit_probability)  # 预处理 位乘
        # 打击概率转换一下：num_target x num_base ---> 1 x num_target

        for i in range(self.num_target):
            self.damage_probability_sum[0][i] = sum(self.damage_probability[i, :])
        # 修改了算法原来用的乘法，改乘加法（中间的
        # damage_expect_arr [1, num_target]
        damage_expect_arr = self.threat_value * self.damage_probability_sum
        # 扩充成[num_weapon, num_target]形状：每行进行复制
        damage_expect = np.tile(damage_expect_arr, (num_weapon, 1))
        # damage_expect = (self.beta * self.threat_value) + (self.gamma * damage_probability_sum)
        # 位乘
        self.expect = expect_temp1 * damage_expect

        # 算法参数设定
        self.t = 1
        self.num_pop = 50  # 群体规模
        self.iter_max = 100
        # self.num_pop = max(5, self.num_target)  # 灰狼群体规模
        # self.iter_max = min(int(3 * self.MAX), 50)
        self.alpha = 0.5
        self.eta = 0.02  # 精英个体比例
        self.delta_p = 1
        self.lamda = 0.6
        self.plan_top = np.full((3, num_weapon), -1)
        self.plan = np.full((self.num_pop, self.num_weapon), -1)
        # self.plan_alpha = np.zeros(num_weapon)
        # self.plan_beta = np.zeros(num_weapon)
        # self.plan_delta = np.zeros(num_weapon)
        self.plan_alpha_fitness = 0
        self.plan_beta_fitness = 0
        self.plan_delta_fitness = 0
        # 适应度保存
        self.fitness = np.zeros(self.num_pop)                             # [1, self.num_pop]
        self.fitness_save = np.zeros((self.iter_max+1, self.num_pop))     # [self.iter_max+1, self.num_pop]
        self.jump1 = True
        self.jump2 = True

    def rand_init_plan(self):
        # 初始化
        for i in range(self.num_pop):
            self.plan[i, :] = self.random_plan()
            # print('初始化第{}个方案为{}'.format(i, self.plan[i, :]))

    # 随机分配方案，还是依照可行性矩阵进行
    def random_plan(self):

        # 存储选中的数的位置
        num_rows = self.feasibility.shape[0]
        num_cols = self.feasibility.shape[1]
        selected_numbers = np.zeros(min(num_rows, num_cols))
        selected_locations = []
        # 对每一行进行操作
        count0 = 0
        while True:
            random_indices = np.zeros(num_rows)
            if num_rows > num_cols:  # 行数大于列数
                # 生成随机索引列表, 从行数中选列数个行，random_indices保存的是选择的行数的位置
                random_indices = np.random.choice(num_rows, num_cols, replace=False)
                # print("random_indices", random_indices)
                random_indices = np.array(sorted(random_indices))
                # print("random_indices", random_indices)
                # 根据索引获取对应的行
                # new_matrix = matrix[random_indices]
                # print("new_matrix", new_matrix)

            elif num_rows == num_cols:  # 行数等于列数
                random_indices = np.array([x for x in range(num_rows)])
                # np.random.shuffle(random_indices)

            else:  # 行数小于列数
                random_indices = np.array([x for x in range(num_rows)])
                # np.random.shuffle(random_indices)

            # col_select_list = np.array([x for x in range(num_cols)])
            # col_select_list：选择列数的个数
            col_select_list = np.array([x for x in range(num_cols)])
            for index, element in enumerate(random_indices):
                # 用可行性矩阵选太慢了
                # row = self.feasibility[element]
                row = self.f[element]
                # 随机选择一个不为0的数，并确保它来自不同的列
                rand_col = np.random.choice(col_select_list)  # 从列数的列表中随机选一列
                # 确定选的元素，将元素值保存在selected_numbers中
                selected_numbers[index] = row[rand_col]
                if row[rand_col] == 0:
                    continue
                else:
                    selected_locations.append((element, rand_col))
                    # 删除候选列表中的列值，下一次不会所有到
                    col_select_list = np.delete(col_select_list, np.where(col_select_list == rand_col))
                if len(col_select_list) == 0:
                    break
            count0 = count0 + 1
            # print("count = ", count)
            if count0 < 50:
                if 0 in selected_numbers:
                    # print('有0元素，返回最开始循环')
                    # 清空保存 因为selected_numbers里面一直会有0，会一直进这个循环
                    continue
                else:
                    break
            else:
                self.jump1 = False
                break

        # 创建一个空列表来存储选中的数
        selected_numbers_list = np.full(self.num_weapon, -1)
        if self.jump1:
            # 将选中的数添加到列表中
            for row, col in selected_locations[-1 * self.num_weapon:]:
                selected_numbers_list[row] = col

        # print("随机产生的方案是{}".format(selected_numbers_list))
        '''

        selected_numbers_list = np.full(self.num_weapon, -1)
        mar0 = self.hit_probability * self.feasibility
        tmp0 = np.tile(self.damage_probability_sum, (self.num_weapon, 1))   # 复制打击概率成n行
        tmp1 = np.tile(self.threat_value, (self.num_weapon, 1))         # 威胁值概率成n行
        # mar1 = tmp0 * self.feasibility
        mar1 = tmp0 * 1
        # mar2 = tmp1 * self.feasibility
        mar2 = tmp1 * 1
        randint = random.randint(0, 3)
        if randint == 0:
            for k in range(self.num_weapon):
                max_index1 = np.argmax(mar0)
                max_row1, max_col1 = np.unravel_index(max_index1, mar0.shape)
                selected_numbers_list[max_row1] = max_col1
                mar0[max_row1, :] = -1
                mar0[:, max_col1] = -1
        elif randint == 1:
            for k in range(self.num_weapon):
                max_index1 = np.argmax(mar1)
                max_row1, max_col1 = np.unravel_index(max_index1, mar0.shape)
                selected_numbers_list[max_row1] = max_col1
                mar0[max_row1, :] = -1
                mar0[:, max_col1] = -1
        else:
            for k in range(self.num_weapon):
                max_index1 = np.argmax(mar2)
                max_row1, max_col1 = np.unravel_index(max_index1, mar0.shape)
                selected_numbers_list[max_row1] = max_col1
                mar0[max_row1, :] = -1
                mar0[:, max_col1] = -1
        '''

        return selected_numbers_list


    # 计算整个群体的适应度，并对方案进行降序
    def cal_fitness(self):
        for n in range(self.num_pop):
            single_plan = []
            single_plan = self.plan[n, :]
            asset_value_sum = self.cal_single_fitness(single_plan)
            self.fitness[n] = asset_value_sum

        # 取出最大适应度值
        # print('各方案为：{}'.format(self.plan))
        # print('各方案适应度值为：{}'.format(self.fitness))
        # 保存当前轮
        self.fitness_save[self.t-1, :] = self.fitness
        # 从大到小排序，确定前三头狼(方案)，保存其方案，对应适应度，对应种群位置——放到不能操作的表里
        fitness_sorted = sorted(self.fitness, reverse=True)
        fitness_list = self.fitness.tolist()
        sort_indices = []

        # 按照适应度从大到小的顺序，搜索原来在fitness的位置，保存在sort_indices中
        for index, value in enumerate(fitness_sorted):
            sort_indices.append(fitness_list.index(value))
        sort_values = [value for value in fitness_sorted[:3]]

        # 重新进行排序
        plan_copy = self.plan
        sort_plan = np.zeros((self.num_pop, self.num_weapon), dtype=np.int64)
        for k in range(self.num_pop):
            sort_plan[k, :] = plan_copy[sort_indices[k], :]

        self.plan = sort_plan
        self.plan_top[0, :] = plan_copy[sort_indices[0], :]
        self.plan_top[1, :] = plan_copy[sort_indices[1], :]
        self.plan_top[2, :] = plan_copy[sort_indices[2], :]

        self.plan_alpha_fitness = sort_values[0]
        self.plan_beta_fitness = sort_values[1]
        self.plan_delta_fitness = sort_values[2]

    '''
    def cal_single_fitness(self, single_plan):
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
                # 对应造成基地损失
                asset_value[0][target_base] = asset_value[0][target_base] * (1 - p2)

                # 最大化消灭武器目标威胁
                value1 = self.hit_probability[m][single_plan[m]]
                value2 = self.threat_value[0][single_plan[m]]
                value[0][m] = value1 * value2

        a = np.array(asset_value).flatten().tolist()    # 转为只有一个中括号的
        b = sum(a)
        value_sum = np.sum(value)
        asset_value_sum = 0.5 * (b/self.asset_value_sum) + 0.5 * (value_sum/self.threat_value_sum)
        # print('计算适应度的方案为：{}'.format(single_plan))
        # print('计算得到的适应度为：{}'.format(asset_value_sum))
        # single_plan_mar = np.zeros((self.num_weapon, self.num_target))
        # for m in range(len(single_plan)):
        #     if single_plan[m] != -1:
        #         single_plan_mar[m][single_plan[m]] = 1
        # 使用打击期望
        # damage_value = single_plan_mar * self.expect
        # damage_value_sum = damage_value.sum()
        # asset_value_sum = self.asset_value_sum - damage_value_sum
        # 使用打击概率
        # damage_value = single_plan_mar * self.hit_probability
        # damage_value_sum = damage_value.sum()
        # asset_value_sum = damage_value_sum
        # 使用威胁值
        # threat_value = np.tile(self.threat_value, (self.num_weapon, 1))
        # damage_value = single_plan_mar * threat_value
        # damage_value_sum = damage_value.sum()
        # asset_value_sum = damage_value_sum

        return asset_value_sum
        '''

    def modular_position_update(self):
        for j in range(4, self.num_pop):
            a = 1 - self.t / self.iter_max
            # print('进入模块化a的值为{}'.format(a))
            rand = random.random()
            # 判定：分别使用轮盘选择和随机分配方案
            if rand <= a:  # 用轮盘选择法
                w = np.zeros(3)
                w[0] = self.plan_alpha_fitness / (
                            self.plan_alpha_fitness + self.plan_beta_fitness + self.plan_delta_fitness)
                w[1] = self.plan_beta_fitness / (
                            self.plan_alpha_fitness + self.plan_beta_fitness + self.plan_delta_fitness)
                w[2] = self.plan_delta_fitness / (
                            self.plan_alpha_fitness + self.plan_beta_fitness + self.plan_delta_fitness)
                cum = 0
                r0 = random.random()
                for k in range(3):
                    cum = cum + w[k]
                    if r0 <= cum:
                        self.plan[j, :] = self.plan_top[k, :]
                        break  # 退出for循环

            else:  # 随机重新分配 但是得满足可行性矩阵
                # 列数大于等于行数
                self.plan[j, :] = self.random_plan()

    def local_search(self):
        # 对头狼进行更新：本地搜索
        for j in range(0, 3):
            plan_base = np.zeros((3, self.num_weapon))
            fit_top = np.zeros(3)
            rand = random.random()
            plan_base[j, :] = self.plan_top[j, :]
            count1 = 0
            if rand <= self.lamda:
                while True:
                    # LSA更新：由于数量都是一个：发射或者不发射-->随机选择可行性矩阵的两个不为0元素，
                    # 如何转换为方案里两个位置的交换：先选方案里两个位置进行交换，然后查看交换位置后两个元素是否满足可行性矩阵
                    # 选出位置和对于值
                    rs1 = random.sample(range(0, self.num_weapon), 2)
                    rs_tmp = map(lambda x: int(x), rs1) #转换数据类型
                    rs = list(rs_tmp)
                    ds = np.array([self.plan_top[j, rs[0]], self.plan_top[j, rs[1]]])
                    ds = ds.astype(int)
                    count1 = count1 + 1
                    # 不都为-1，且交换后可行性矩阵满足，进行交换
                    if count1 < 30:
                        if -1 not in ds and (self.feasibility[rs[0]][ds[1]] * self.feasibility[rs[1]][ds[0]] == 1):
                            plan_base[j, rs[0]] = ds[1]
                            plan_base[j, rs[1]] = ds[0]
                            break  # 交换完，退出循环

                        # 有一个为1， 交换后可行性矩阵满足: 但是可能寻址不满足
                        elif (-1 in ds) and (sum(ds) != -2):
                            if ds[0] == -1 and self.feasibility[rs[0]][ds[1]] == 1:
                                plan_base[j, rs[0]] = ds[1]
                                plan_base[j, rs[1]] = ds[0]
                                break
                            if ds[1] == -1 and self.feasibility[rs[1]][ds[0]] == 1:
                                plan_base[j, rs[0]] = ds[1]
                                plan_base[j, rs[1]] = ds[0]
                                break
                        else:  # 都是-1：重新进行选择
                            continue
                    else:
                        self.jump2 = False
                        break
                # 计算操作后的适应度，用大的替换原有的方案
        fit_top[j] = self.cal_single_fitness(plan_base[j, :])
        if fit_top[j] > self.fitness[j]:
            self.plan[j, :] = plan_base[j, :]
            # print('替换后的方案为{}'.format(self.plan[j, :]))

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

    # 整个搜索过程 包含模块化位置更新和本地更新
    def search(self):
        if self.jump1 and self.jump2:
            # 迭代次数
            for i in range(self.iter_max):
                self.cal_fitness()
                # 模块化更新位置：对象是除了头三只和精英个体外的所有个体
                self.modular_position_update()
                if self.num_weapon > 1:
                    self.local_search()
                # print("第{}次迭代最好方案有：{}".format(i, self.plan[0, :]))
                self.t = self.t + 1

                # 取出每轮迭代最好方案，分别计算俩个目标函数
                iter_plan = self.plan[0, :]
                p1, p2 = self.cal_iter(iter_plan)
                # print('剩余基地价值比例为：{}'.format(p1))
                # print('预计打击目标比例为：{}'.format(p2))
                '''
                with open('data_HDGWO_0.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([p1, p2])
                
                '''




    def run(self):
        self.rand_init_plan()
        self.search()
        self.cal_fitness()  # 最后一个搜索，再进行一次计算适应度和排序
        # print("最后一次迭代方案一共有：{}".format(self.plan))

        best_plan = self.plan[0, :].astype('int32')
        best_plan = best_plan.tolist()
        # print("最后输出方案为{}".format(best_plan))
        # print("可行性矩阵为{}".format(self.feasibility))
        return best_plan
























        

