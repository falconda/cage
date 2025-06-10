import random
import numpy as np
import copy
import csv

class DDEMRR_WTA(object):
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
        self.feasibility1 = np.ones((num_weapon, num_target))
        for i in range(self.num_target):
            self.damage_probability_sum[0][i] = sum(self.damage_probability[i, :])
        # 目标打击基地的列表
        self.Tar_to_Asset = np.nonzero(self.damage_probability)[1]

        # 算法参数设定
        self.t = 1
        self.elite = 15
        self.num_pop = 50  # 群体规模
        self.iter_max = 30
        # self.num_pop = 30  # 群体规模
        # self.iter_max = 30
        # self.num_pop = max(5, self.num_target)  # 群体规模
        # self.iter_max = min(int(3 * self.num_target), 100)
        self.plan = np.full((self.num_pop, self.num_weapon), -1)
        self.f = 0.7        # 缩放因子，用于差分选择位置
        self.p_de = 0.9     # 交叉操作概率
        self.alpha = 0.5    # 用于目标函数加权
        self.rng = np.random.default_rng(123)  # 用于产生随机数
        self.MIN = min(num_weapon, num_target)   # 注意分三种情况讨论
        # 适应度保存
        self.fitness = np.zeros(self.num_pop)  # [1, self.num_pop]
        self.fitness_save = np.zeros((self.iter_max + 1, self.num_pop))  # [self.iter_max+1, self.num_pop]

        self.jump1 = True

    def rand_init_plan(self):
        # 初始化
        for i in range(self.num_pop):
            self.plan[i, :] = self.random_plan()
            # print('初始化第{}个方案为{}'.format(i, self.plan[i, :]))

    # 随机分配方案，还是依照可行性矩阵进行
    def random_plan(self):
        pos = self.rng.uniform(0, 1, self.MIN)
        # 相等：就是为TSP问题，直接用最小位置匹配值法
        dict_list = {}
        for index, values in enumerate(sorted(pos)):
            dict_list[values] = index
        result_list = []
        for i in pos:
            result_list.append(dict_list[i])
        if self.num_weapon == self.num_target:
            plan = np.array(result_list)
        # 武器数量多，要添加-1
        elif self.num_weapon > self.num_target:
            # 先创建num_target长度的实数向量
            plan_arr = np.array(result_list)
            plan = copy.deepcopy(plan_arr)
            # 填充-1
            gap = self.num_weapon - self.num_target # 确定个数
            for i in range(gap):
                index = random.randint(0, plan.shape[0])
                plan = np.insert(plan, index, -1)

        else:   # 目标数量多：序列不会是连续的整数
                # 在num_target个数中选取num_target个数，按照最小位置匹配值的顺序排列
            arr = [x for x in range(self.num_target)]
            selected_item = random.sample(arr, self.num_weapon)
            selected_item_sort = np.sort(selected_item)
            plan_list = []
            for i in range(self.num_weapon):
                plan_list.append(selected_item_sort[result_list[i]])
            plan = np.array(plan_list)
        return plan


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

    #  离散差分变异
    def differential_mutation(self, plan):
        plan_select = copy.deepcopy(plan)  # 确定选择的方案
        # 选取两个不同的plan，对当前plan进行差分变异
        rand1 = random.randrange(0, self.num_pop)
        rand2 = random.randrange(0, self.num_pop)
        plan1 = self.plan[rand1, :]
        plan2 = self.plan[rand2, :]
        # 去除-1的值，长度会变，没有-1的长度:武器数量小于目标数量，可能长度会不一样，怎么办？
        if self.num_weapon <= self.num_target:
            plan1_slice = np.delete(plan1, np.where(plan1 == -1))
            plan2_slice = np.delete(plan1, np.where(plan2 == -1))
            length_min = self.num_weapon
        else:   # 武器数量大于目标数量，长度不一样,对齐最短的
            plan1_slice = np.delete(plan1, np.where(plan1 == -1))
            plan2_slice = np.delete(plan1, np.where(plan2 == -1))
            length_1 = len(plan1_slice)
            length_2 = len(plan2_slice)
            length_3 = len(plan_select)
            length_min = min(length_1, length_2, length_3)
            plan1_slice = plan1_slice[:length_min]
            plan1_slice = plan1_slice[:length_min]

        plan_alpha = plan1_slice - plan2_slice
        # print("plan1_slice is {}".format(plan1_slice))
        # print("plan2_slice is {}".format(plan2_slice))
        # print("plan_alpha is {}".format(plan_alpha))
        # print("plan_select is {}".format(plan_select))

        random_arr = np.random.rand(length_min)  # 确定每个位置差分的概率
        plan_beta = np.zeros(length_min)  # 保存确定差分的位置的值
        # 选择差分的位置
        for j in range(length_min):
            if random_arr[j] < self.f:
                plan_beta[j] = plan_alpha[j]
        plan_slice = np.delete(plan_select, np.where(plan_select == -1))
        plan_slice = plan_slice[:length_min]
        # plan_slice是选择的方案（去掉-1）,plan_beta是差分算法，进行差分运算, plan_dm是运算得到的结果（带重复）
        plan_dm = np.zeros(length_min)
        for k in range(length_min):
            x = plan_slice[k]
            delta = plan_beta[k]
            # 似乎这么计算需要考虑武器和目标的大小关系？有待考证
            # 武器数量>目标数量：删掉-1就是一样的：维度=武器数量
            # 武器数量<目标数量：删掉-1后，看作：维度=武器数量的其中一部分，也是可以的
            plan_dm[k] = np.mod((x + delta + self.num_target - 1), self.num_target)

        # 处理重复的部分:记录重复出现的位置，并填上不同的数字
        plan_list = plan_dm.astype('int32').tolist()
        elements = []  # 收集已经出现的元素
        duplicates = np.zeros(length_min)  # 记录重复出现的位置 用1表示
        for m in range(length_min):
            if plan_list[m] not in elements:
                elements.append(plan_list[m])
            else:
                duplicates[m] = 1
        # print("duplicates are {}".format(duplicates))
        # print("elements are {}".format(elements))
        weapon_arr = np.arange(length_min)
        for n in range(len(elements)):
            indices = np.where(weapon_arr == elements[n])
            weapon_arr = np.delete(weapon_arr, indices)
        # print("weapon_arr is {}".format(weapon_arr))
        # 重复的位置选择elements以外的元素
        # selected_elements = np.array(weapon_arr)
        count = np.sum(duplicates).astype('int32')  # 确定重复个数，也就是重新替换的个数
        selected_elements = np.random.choice(weapon_arr, size=count, replace=False)

        # print("selected_elements is {}".format(selected_elements))
        # 对于有duplicates中有1的位置，用selected_elements中的元素对duplicates进行填补，填补成不重复的形式
        plan_de = np.zeros(len(duplicates))
        for l in range(len(duplicates)):
            if duplicates[l] == 0:
                plan_de[l] = plan_dm[l]
            elif duplicates[l] == 1:  # 如何填补成不重复
                plan_de[l] = np.random.choice(selected_elements, size=1, replace=False)
                indices = np.where(selected_elements == plan_de[l])
                selected_elements = np.delete(selected_elements, indices)
        # print("plan_de is {}".format(plan_de))

        # 操作结束，如果武器数量大于目标数量，要把-1还原回去（一开始删掉的）
        p = 0
        plan_expand = np.zeros(len(plan))
        for t in range(len(plan)):
            if plan[t] != -1:
                plan_expand[t] = plan_de[p]
                p = p + 1
            else:   # 武器没有选上目标，即为-1
                plan_expand[t] = -1
        np.random.shuffle(plan_expand)  # 随机洗乱方案
        # greedy选取方案
        fitness_done = self.cal_single_fitness(plan_expand.astype('int32'))
        fitness = self.cal_single_fitness(plan_select)
        if fitness_done > fitness:
            plan_done = plan_expand
        else:
            plan_done = self.random_plan()
        return plan_done.astype('int32')

    def crossover_operations(self, plan):
        rand = random.random()
        plan_base = copy.deepcopy(plan)
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
                    if -1 not in ds and (self.feasibility1[rs[0]][ds[1]] * self.feasibility1[rs[1]][ds[0]] == 1):
                        plan_base[rs[0]] = ds[1]
                        plan_base[rs[1]] = ds[0]
                        break  # 交换完，退出循环

                    # 有一个为1， 交换后可行性矩阵满足: 但是可能寻址不满足
                    elif (-1 in ds) and (sum(ds) != -2):
                        if ds[0] == -1 and self.feasibility1[rs[0]][ds[1]] == 1:
                            plan_base[rs[0]] = ds[1]
                            plan_base[rs[1]] = ds[0]
                            break
                        if ds[1] == -1 and self.feasibility1[rs[1]][ds[0]] == 1:
                            plan_base[rs[0]] = ds[1]
                            plan_base[rs[1]] = ds[0]
                            break
                    else:  # 都是-1：重新进行选择
                        continue

            # 计算操作后的适应度，用大的替换原有的方案
            fitness_done = self.cal_single_fitness(plan_base.astype('int32'))
            fitness = self.cal_single_fitness(plan.astype('int32'))
            if fitness_done > fitness:
                plan_done = plan_base
            else:
                plan_done = plan_base
            # print('替换后的方案为{}'.format(self.plan[j, :]))
        else:
            # 受体编辑 随机选择一个片段进行倒序，再放回
            arr = [n for n in range(self.num_weapon)]
            location = random.sample(arr, 2)
            location = np.array(sorted(location))  # 确定开始位置和结束位置
            arr_clip = plan_base[location[0]:location[1]]
            arr_clip_re = arr_clip[::-1]
            plan_base[location[0]:location[1]] = arr_clip_re
            fitness_done = self.cal_single_fitness(plan_base.astype('int32'))
            fitness = self.cal_single_fitness(plan.astype('int32'))
            if fitness_done > fitness:
                plan_done = plan_base
            else:
                plan_done = plan_base
        return plan_done.astype('int32')

    # 整个搜索过程
    def search(self):
        # 迭代次数
        for i in range(self.iter_max):
            self.cal_fitness()  # 保存了前3个个体的位置

            iter_plan = self.plan[0, :]
            p1, p2 = self.cal_iter(iter_plan)
            # print('剩余基地价值比例为：{}'.format(p1))
            # print('预计打击目标比例为：{}'.format(p2))

            # 保存数据到csv文件
            '''
            with open('data_DDE_0.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([p1, p2])            
            '''

            if self.num_weapon > 1:
                for j in range(3, self.elite):
                    plan_selected = self.plan[j, :]
                    plan_dm = self.crossover_operations(plan_selected)
                    self.plan[j, :] = plan_dm
                for k in range(self.elite, self.num_pop):
                    plan_selected = self.plan[k, :]
                    plan_de = self.differential_mutation(plan_selected)
                    self.plan[k, :] = plan_de

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