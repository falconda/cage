import random
import numpy as np
import copy
import csv

class FSA_WTA(object):
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
        self.t = 1
        self.num_pop = 50  # 群体规模
        self.iter_max = 100
        self.Pb = 0.1
        self.alpha = 0.5    # 用于目标函数加权
        self.rng = np.random.default_rng(123)  # 用于产生随机数
        self.MIN = min(num_weapon, num_target)   # 注意分三种情况讨论
        self.position = np.zeros((self.num_pop, self.MIN))
        self.plan = np.full((self.num_pop, self.num_weapon), -1)
        self.plan_one = np.full((self.iter_max, self.num_weapon), -1)
        # 适应度保存
        self.fitness = np.zeros(self.num_pop)  # [self.iter_max+1, self.num_pop]

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

    # 更新步骤，按照适应度进行排序
    def update(self, position, plan):
        length = plan.shape[0]
        fitness = np.zeros(length)
        for n in range(length):
            single_plan = plan[n, :]
            fit = self.cal_single_fitness(single_plan)
            fitness[n] = fit
        fitness_sort = sorted(fitness, reverse=False)
        self.fitness = np.array(fitness_sort)
        fitness_list = fitness.tolist()
        sort_indices = []
        # 按照适应度从大到小的顺序，搜索原来在fitness的位置，保存在sort_indices中
        for index, value in enumerate(fitness_sort):
            sort_indices.append(fitness_list.index(value))
        # 依据适应度对整个种群重新排序
        plan_copy = copy.deepcopy(self.plan)
        sort_plan = np.zeros((self.num_pop, self.num_weapon), dtype=np.int64)
        for k in range(self.num_pop):
            sort_plan[k, :] = plan_copy[sort_indices[k], :]
        self.plan = sort_plan


    def rand_init_pos(self):
        # 初始化
        for i in range(self.num_pop):
            self.position[i, :] = self.random_pos()
            # print('初始化第{}个方案为{}'.format(i, self.plan[i, :]))

    # 产生实数向量，内容是0-1的向量
    def random_pos(self):
        pos = np.random.rand(self.MIN)
        return pos

    # 迁徙操作
    def migration(self, position, best_position):
        omega = self.rng.normal(loc=0, scale=1.2, size=1)
        migration_position = position + omega * (best_position - position)
        return migration_position

    # 觅食操作
    def foraging(self, position, best_position):
        a = random.random()
        if a > 0.5:
            e1 = 1
        else:
            e1 = -1
        b = random.random()
        if b > 0.5:
            e2 = 1
        else:
            e2 = -1
        foraging_position = (position + e1 * best_position + np.random.randn() *
                             np.abs(np.random.randn() * best_position + e2 * position))/(self.rng.chisquare(df=8, size=1))

        return foraging_position

    # 实数向量转换为整数向量的方案
    # 最重要的问题: pos长度：min(num_weapon, num_target)
    def transformation(self, pos):
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

    # 对于整数向量：进行局部搜索操作：倒序，洗乱，交叉
    def local_search(self, plan):
        rand = self.rng.integers(3)
        plan_base = copy.deepcopy(plan)
        if rand == 1:   # 进行交叉操作
            rs = np.array(random.sample(range(0, self.num_weapon), 2))
            ds = np.array([plan_base[rs[0]], plan_base[rs[1]]])
            plan_base[rs[0]] = ds[1]
            plan_base[rs[1]] = ds[0]
            # 计算操作后的适应度，用大的替换原有的方案
            fitness_done = self.cal_single_fitness(plan_base.astype('int32'))
            fitness = self.cal_single_fitness(plan.astype('int32'))
            if fitness_done > fitness:
                plan_done = plan_base
            else:
                plan_done = plan
            # print('替换后的方案为{}'.format(self.plan[j, :]))
        elif rand == 2:
            # 受体编辑 随机选择一个片段进行倒序，再放回
            arr = [n for n in range(self.num_weapon)]
            location = random.sample(arr, 2)
            location = np.array(sorted(location))  # 确定开始位置和结束位置
            arr_clip = plan_base[location[0]:location[1]]   # 切片
            arr_clip_re = arr_clip[::-1]    # 倒序
            plan_base[location[0]:location[1]] = arr_clip_re
            fitness_done = self.cal_single_fitness(plan_base.astype('int32'))
            fitness = self.cal_single_fitness(plan.astype('int32'))
            if fitness_done > fitness:
                plan_done = plan_base
            else:
                plan_done = plan
        else:    # 洗乱操作
            arr = [n for n in range(self.num_weapon)]
            location = random.sample(arr, 2)
            location = np.array(sorted(location))  # 确定开始位置和结束位置
            arr_clip = plan_base[location[0]:location[1]]   # 切片
            random.shuffle(arr_clip)
            plan_base[location[0]:location[1]] = arr_clip
            fitness_done = self.cal_single_fitness(plan_base.astype('int32'))
            fitness = self.cal_single_fitness(plan.astype('int32'))
            if fitness_done > fitness:
                plan_done = plan_base
            else:
                plan_done = plan
        plan_search = plan_done
        return plan_search

    # 整个搜索过程
    def search(self):
        # 确定种群里的操作的数量：觅食操作，迁徙操作，移居操作？
        for n in range(self.iter_max):
            # 更新步骤，计算每个个体的适应度并且进行(升序)排序
            best_position = self.position[self.num_pop-1, :]
            R = random.random()
            # 确定各部分数量
            MPr = int(R * self.num_pop * (1 - self.Pb))
            MPo = int(self.Pb * self.num_pop)
            MPt = self.num_pop - MPr - MPo
            for i in range(MPo):
                position = self.position[i, :]
                migration_position = self.migration(position, best_position)
                plan = self.transformation(migration_position)
                self.position[i, :] = migration_position
                self.plan[i, :] = plan
            for j in range(MPo, MPo+MPt):
                position = self.position[j, :]
                foraging_position = self.foraging(position, best_position)
                plan = self.transformation(foraging_position)
                self.position[j, :] = foraging_position
                self.plan[j, :] = plan
            for k in range(MPo+MPt, self.num_pop):
                position = self.position[k, :]
                migration_position = self.migration(position, best_position)
                plan = self.transformation(migration_position)
                self.position[k, :] = migration_position
                self.plan[k, :] = plan
            self.update(self.position, self.plan)

            # 取最优的个体，进行局部搜索
            plan_mid = self.plan[self.num_pop-1, :]  # 更新完放入
            if self.num_weapon > 1:
                for j in range(15):
                    plan_search = self.local_search(plan_mid)
            else:
                plan_search = plan_mid
            # 历史迭代，选择相比于之前更优秀的
            if n == 0:
                plan_iter = plan_search
            else:
                plan_pre = self.plan_one[n - 1, :]
                fit_now = self.cal_single_fitness(plan_search)
                fit_pre = self.cal_single_fitness(plan_pre)
                if fit_now > fit_pre:
                    plan_iter = plan_search
                else:
                    plan_iter = plan_pre
            self.plan[self.num_pop-1, :] = plan_iter
            # 保存每次迭代最优个体
            self.plan_one[n, :] = plan_iter
            # 适应度是由低到高进行排序
            self.t = self.t + 1
            # 迭代性能计算
            p1, p2 = self.cal_iter(plan_iter)
            # print('剩余基地价值比例为：{}'.format(p1))
            # print('预计打击目标比例为：{}'.format(p2))
            # 保存数据到csv文件

            '''
            with open('data_FSA_0.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([p1, p2])             
            
            '''


    def run(self):
        # 初始化群体位置
        self.rand_init_pos()
        # 产生对应方案
        for m in range(self.num_pop):
            position = self.position[m, :]
            plan = self.transformation(position)
            self.plan[m, :] = plan
        # 进行升序排序，最小的在前面
        self.update(self.position, self.plan)
        # 算法群体更新
        self.search()
        # 选择最优的种马个体
        best_plan = self.plan_one[self.iter_max-1, :].astype('int32').tolist()
        # print("输出方案为{}".format(best_plan))
        # print("可行性矩阵为{}".format(self.feasibility))
        return best_plan