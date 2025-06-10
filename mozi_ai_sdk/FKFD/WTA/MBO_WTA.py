import random
import numpy as np
import copy
import csv

class MBO_WTA(object):
    def __init__(self, num_weapon, num_target, Wei, Pij, Fij, Qjk, V_a):
        self.num_weapon = num_weapon
        self.num_target = num_target
        self.threat_value = Wei  # [1, num_target]
        self.hit_probability = Pij  # [num_weapon, num_target]
        self.feasibility = Fij  # [num_weapon, num_target]
        self.damage_probability = Qjk  # [num_target, num_base],num_base=3，并且每一行只有一个元素不为0
        self.asset_value = V_a  # [1, num_base]， num_base=3
        self.asset_value_sum = sum(V_a[0])
        self.threat_value_sum = sum(Wei[0])

        # 目标打击基地的列表
        self.Tar_to_Asset = np.nonzero(self.damage_probability)[1]
        self.MAX = max(num_weapon, num_target)
        self.damage_probability_sum = np.zeros((1, self.num_target))
        self.f = np.ones((num_weapon, num_target))
        for i in range(self.num_target):
            self.damage_probability_sum[0][i] = sum(self.damage_probability[i, :])
        # 目标打击基地的列表
        self.Tar_to_Asset = np.nonzero(self.damage_probability)[1]

        # 算法参数设定
        self.t = 1
        self.elite = 2
        self.num_pop = 80  # 帝王蝶群体规模
        self.iter_max = 100
        # self.num_pop = 30  # 帝王蝶群体规模
        # self.iter_max = 30
        # self.num_pop = max(5, self.num_target)  # 帝王蝶群体规模
        # self.iter_max = min(int(3 * self.num_target), 100)
        self.plan = np.full((self.num_pop, self.num_weapon), -1)
        self.alpha = 0.5
        self.p1 = 0.5   # 划分群体概率
        self.p2 = 0.5   # NP2操作概率
        self.p_de = 0.5
        np1_num = 30
        self.np1 = np.zeros((np1_num, self.num_weapon), dtype=int)
        self.np2 = np.zeros((self.num_pop - np1_num, self.num_weapon), dtype=int)

        # 适应度保存
        self.fitness = np.zeros(self.num_pop)  # [1, self.num_pop]
        self.fitness_save = np.zeros((self.iter_max + 1, self.num_pop))  # [self.iter_max+1, self.num_pop]
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
            if count0 < 20:
                if 0 in selected_numbers:
                    # print('有0元素，返回最开始循环')
                    # 清空保存 因为selected_numbers里面一直会有0，会一直进这个循环
                    continue
                else:
                    break
            else:
                self.jump1 = False
                break
        selected_numbers_list = np.full(self.num_weapon, -1)
        if self.jump1:
            # 将选中的数添加到列表中
            for row, col in selected_locations[-1 * self.num_weapon:]:
                selected_numbers_list[row] = col

        return selected_numbers_list

    # 适应度计算
    def cal_fitness(self):
        for n in range(self.num_pop):

            single_plan = self.plan[n, :]
            asset_value_sum = self.cal_single_fitness(single_plan)
            self.fitness[n] = asset_value_sum
        # 保存当前轮
        self.fitness_save[self.t - 1, :] = self.fitness
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

    # 分成两个群体：NP1和NP2
    def deliver_2pop(self):
        np1_num = 30
        np2_num = self.num_pop - np1_num
        for i in range(0, np2_num):
            self.np2[i, :] = self.plan[i, :]
        for j in range(np2_num, self.num_pop):
            self.np1[j - np2_num, :] = self.plan[j, :]
        '''
        self.np1 = np.row_stack((self.np1, self.plan[0, :]))
        self.np2 = np.row_stack((self.np2, self.plan[1, :]))
        for i in range(self.num_pop-2):
            rand = random.random()
            if rand < self.p1:
                self.np1 = np.row_stack((self.np1, self.plan[i+2, :]))
            else:
                self.np2 = np.row_stack((self.np2, self.plan[i+2, :]))
        self.np1 = np.delete(self.np1, 0, axis=0)   # 删除第一行
        self.np2 = np.delete(self.np2, 0, axis=0)
        # print('NP1群体为{}'.format(self.np1.shape))
        # print('NP2群体为{}'.format(self.np2.shape))
        '''


    # NP1更新：greedy随机从NP1和NP2中选
    def migration_operations(self):
        for i in range(len(self.np1)):
            a = 1 - self.t / self.iter_max
            rand = random.random()
            if rand < a:# 从np1里选
                '''
                if len(self.np1) > 0:
                    location = np.random.randint(0, len(self.np1), 1)

                    select = []
                    select = self.np1[location, :]
                    select_fitness = self.cal_single_fitness(select)
                    if select_fitness > self.cal_single_fitness(self.np1[i, :]):
                        self.np1[i, :] = self.np1[location, :]  # 进行更新
                '''
                if len(self.np1) > 0:
                    self.np1[i, :] = self.plan[0, :]


            else:
                '''
                if len(self.np2) > 0:
                    location = np.random.randint(0, len(self.np1), 1)
                    select = []
                    select = self.np2[location, :]
                    select_fitness = self.cal_single_fitness(select)
                    if select_fitness > self.cal_single_fitness(self.np1[i, :]):
                        self.np1[i, :] = self.np2[location, :]  # 进行更新
                '''
                if len(self.np1) > 0:
                    self.np1[i, :] = self.random_plan()

    # NP2：随机选best或者随机从NP2中选：用Cr算子进行变异操作
    def adjust_crossover_operations(self):
        for i in range(len(self.np2)):
            rand = random.random()
            plan_base = self.np2[i, :]
            count1 = 0
            if rand <= self.p_de:
                # 随机点更新
                while True:
                    # LSA更新：由于数量都是一个：发射或者不发射-->随机选择可行性矩阵的两个不为0元素，
                    # 如何转换为方案里两个位置的交换：先选方案里两个位置进行交换，然后查看交换位置后两个元素是否满足可行性矩阵
                    # 选出位置和对于值
                    rs1 = random.sample(range(0, self.num_weapon), 2)
                    rs_tmp = map(lambda x: int(x), rs1)  # 转换数据类型
                    rs = list(rs_tmp)
                    ds = np.array([plan_base[rs[0]], plan_base[rs[1]]])
                    ds = ds.astype(int)
                    count1 = count1 + 1
                    # 不都为-1，且交换后可行性矩阵满足，进行交换
                    if count1 < 30:
                        if -1 not in ds and (self.feasibility[rs[0]][ds[1]] * self.feasibility[rs[1]][ds[0]] == 1):
                            plan_base[rs[0]] = ds[1]
                            plan_base[rs[1]] = ds[0]
                            break  # 交换完，退出循环

                        # 有一个为1， 交换后可行性矩阵满足: 但是可能寻址不满足
                        elif (-1 in ds) and (sum(ds) != -2):
                            if ds[0] == -1 and self.feasibility[rs[0]][ds[1]] == 1:
                                plan_base[rs[0]] = ds[1]
                                plan_base[rs[1]] = ds[0]
                                break
                            if ds[1] == -1 and self.feasibility[rs[1]][ds[0]] == 1:
                                plan_base[rs[0]] = ds[1]
                                plan_base[rs[1]] = ds[0]
                                break
                        else:  # 都是-1：重新进行选择
                            continue
                    else:
                        self.jump2 = False
                        break
                # 计算操作后的适应度，用大的替换原有的方案
                fitness_done = self.cal_single_fitness(plan_base)
                fitness = self.cal_single_fitness(self.np2[i, :])
                if fitness_done > fitness:
                    self.np2[i, :] = plan_base
                # print('替换后的方案为{}'.format(self.plan[j, :]))
            else:
                # 受体编辑 随机选择一个片段进行倒序，再放回
                arr = [n for n in range(self.num_weapon)]
                location = random.sample(arr, 2)
                location = np.array(sorted(location))  # 确定开始位置和结束位置
                arr_clip = plan_base[location[0]:location[1]]
                arr_clip_re = arr_clip[::-1]
                plan_base[location[0]:location[1]] = arr_clip_re
                fitness_done = self.cal_single_fitness(plan_base)
                fitness = self.cal_single_fitness(self.np2[i, :])
                if fitness_done > fitness:
                    self.np2[i, :] = plan_base

        '''
         for i in range(len(self.np2)):
            rand = random.random()
            if rand < 0.2:  # 选择适应度最高的个体
                self.np2[i, :] = self.np2[0, :]
                # self.np2[i, :] = self.random_plan()
            else:
                
                # 使用Cr算子进行
                location = np.random.randint(0, len(self.np2), 1)
                self.np2[i, :] = self.np2[location, :]
                a = self.cal_single_fitness(self.np2[i, :])
                b = self.cal_single_fitness(self.plan[0, :])
                dis1 = a - b
                c = self.cal_single_fitness(self.plan[len(self.plan)-1, :])
                d = self.cal_single_fitness(self.plan[0, :])
                dis2 = c - d
                dis = (dis1/dis2)
                
                # rand_de = random.random()
                # if rand_de < dis and self.num_target > 1 and self.num_weapon > 1:
                if self.num_target > 1 and self.num_weapon > 1:   # 使用Cr算子:适应度越低，越有概率进行这个操作
                    plan_base = self.np2[i, :]
                    arr = [n for n in range(self.num_weapon)]
                    location = random.sample(arr, 2)
                    location = np.array(sorted(location))  # 确定开始位置和结束位置
                    arr_clip = plan_base[location[0]:location[1]]
                    arr_clip_re = arr_clip[::-1]
                    plan_base[location[0]:location[1]] = arr_clip_re
                    # fitness_done = self.cal_single_fitness(plan_base)
                    self.np2[i, :] = plan_base
        
        :return: 
        '''

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

    # 整个搜索过程 包含对NP1种群的操作和对NP2种群的操作
    def search(self):
        # 迭代次数
        for i in range(self.iter_max):
            self.cal_fitness()  # 保存了前2个个体的位置

            iter_plan = self.plan[0, :]
            p1, p2 = self.cal_iter(iter_plan)
            # print('剩余基地价值比例为：{}'.format(p1))
            # print('预计打击目标比例为：{}'.format(p2))
            '''
            # 保存数据到csv文件
            with open('data_MBO_0.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([p1, p2])
            '''

            self.deliver_2pop()
            self.migration_operations()
            self.adjust_crossover_operations()
            self.plan = np.row_stack((self.np1, self.np2))
            # print("第{}次迭代最好方案有：{}".format(i, self.plan[0, :]))
            self.t = self.t + 1

    def run(self):
        self.rand_init_plan()
        self.search()
        self.cal_fitness()  # 最后一个搜索，再进行一次计算适应度和排序
        # print("最后一次迭代方案一共有：{}".format(self.plan))

        best_plan = self.plan[0, :].astype('int32')
        best_plan = best_plan.tolist()
        # print("输出方案为{}".format(best_plan))
        # print("可行性矩阵为{}".format(self.feasibility))
        return best_plan