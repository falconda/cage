import random
import numpy as np
import copy
import csv

class SPO_WTA(object):
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
        self.num_pop = 10  # 群体规模
        self.iter_max = 150
        self.alpha = 0.5    # 用于目标函数加权
        self.rng = np.random.default_rng(123)  # 用于产生随机数
        self.MIN = min(num_weapon, num_target)   # 注意分三种情况讨论
        self.position = np.zeros((self.num_pop, self.MIN))
        self.plan = np.full((self.num_pop, self.num_weapon), -1)
        self.plan_one = np.full((self.iter_max, self.num_weapon), -1)
        # 适应度保存
        self.fitness = np.zeros((self.iter_max + 1, self.num_pop))  # [self.iter_max+1, self.num_pop]
        # 信息保存用
        self.pointer1 = False   # 打印两个指标
        self.pointer2 = False    # 保存迭代文件

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
        plan_copy = copy.deepcopy(plan)
        position_copy = copy.deepcopy(position)
        sort_plan = np.zeros((self.num_pop, self.num_weapon), dtype=np.int64)
        sort_position = np.zeros((self.num_pop, self.MIN))
        for k in range(self.num_pop):
            sort_plan[k, :] = plan_copy[sort_indices[k], :]
            sort_position[k, :] = position_copy[sort_indices[k], :]

        return sort_position, sort_plan

    def rand_init_pos(self):
        # 初始化
        for i in range(self.num_pop):
            self.position[i, :] = self.random_pos()
            # print('初始化第{}个方案为{}'.format(i, self.plan[i, :]))

    # 产生实数向量，内容是0-1的向量
    def random_pos(self):
        pos = self.rng.uniform(0, 1, self.MIN)
        return pos

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
        # 先创立分组
        group1_num = int(self.num_pop / 3)
        # group1_position = np.zeros((group1_num, self.MIN))
        # group1_plan = np.full((group1_num, self.num_weapon), -1)
        # group1_fit = np.zeros(group1_num)
        group2_num = int(self.num_pop / 3)
        # group2_position = np.zeros((group2_num, self.MIN))
        # group2_fit = np.full((group2_num, self.num_weapon), -1)
        # group2_plan = np.zeros(group2_num)
        group3_num = self.num_pop - group1_num - group2_num
        # group3_position = np.zeros((group3_num, self.MIN))
        # group3_plan = np.full((group3_num, self.num_weapon), -1)
        # group3_fit = np.zeros(group3_num)
        for n in range(self.iter_max):
            # 赋值
            group1_position = self.position[0:group1_num, :]
            group2_position = self.position[group1_num:(group1_num+group2_num), :]
            group3_position = self.position[(group1_num+group2_num):self.num_pop, :]
            # 四个不同方法确定四个不同pos，plan，fit，选最优
            for i in range(self.num_pop):
                position = self.position[i, :]
                plan = self.plan[i, :]
                fitness = self.fitness[i]
                new_position = np.zeros((4, self.MIN))
                new_plan = np.full((4, self.num_weapon), -1)
                new_fit = np.zeros(4)

                # Complementary combination
                id1 = self.rng.integers(group1_num)
                id2 = self.rng.integers(group2_num)
                rand1 = self.rng.uniform(0, 1, self.MIN)
                new_position[0, :] = position + rand1 * (group1_position[id1, :] - group2_position[id2, :])
                new_plan[0, :] = self.transformation(new_position[0, :])
                new_fit[0] = self.cal_single_fitness(new_plan[0, :])

                # Analog Combination
                if i <= group1_num:
                    id = self.rng.integers(group1_num, size=2)
                    analog_position = group1_position
                elif i <= group1_num+ group2_num:
                    id = self.rng.integers(group2_num, size=2)
                    analog_position = group2_position
                else:
                    id = self.rng.integers(group3_num, size=2)
                    analog_position = group3_position
                rand2 = self.rng.uniform(0, 1, self.MIN)
                new_position[1, :] = position + rand2 * (analog_position[id[1], :] - analog_position[id[0], :])
                new_plan[1, :] = self.transformation(new_position[1, :])
                new_fit[1] = self.cal_single_fitness(new_plan[1, :])

                # Triangle Combination
                id1 = self.rng.integers(group1_num)
                id2 = self.rng.integers(group2_num)
                id3 = self.rng.integers(group3_num)
                rand3 = self.rng.uniform(0, 1, self.MIN)
                rand31 = self.rng.uniform(0, 1, self.MIN)
                rand32 = self.rng.uniform(0, 1, self.MIN)
                rand33 = self.rng.uniform(0, 1, self.MIN)
                new_position[2, :] = position + rand3 * \
                                     ((rand31*group1_position[id1]+rand32*group2_position[id2]+rand33*group3_position[id3])/3)
                new_plan[2, :] = self.transformation(new_position[2, :])
                new_fit[2] = self.cal_single_fitness(new_plan[2, :])

                # Rectangle Combination
                id1 = self.rng.integers(group1_num)
                id2 = self.rng.integers(group2_num)
                id3 = self.rng.integers(group3_num)
                id4 = self.rng.integers(self.num_pop)
                rand4 = self.rng.uniform(0, 1, self.MIN)
                rand41 = self.rng.uniform(0, 1, self.MIN)
                rand42 = self.rng.uniform(0, 1, self.MIN)
                rand43 = self.rng.uniform(0, 1, self.MIN)
                rand44 = self.rng.uniform(0, 1, self.MIN)
                new_position[3, :] = position + rand4*\
                                     ((rand41*group1_position[id1]+rand42*group2_position[id2]+rand43*group3_position[id3]+rand44*self.position[id4, :])/4)
                new_plan[3, :] = self.transformation(new_position[3, :])
                new_fit[3] = self.cal_single_fitness(new_plan[3, :])

                #  选取适应度最大的作为新位置
                index = np.argmax(new_fit)
                new_fit_best = new_fit[index]
                if new_fit_best > fitness:
                    self.position[i, :] = new_position[index, :]
                    self.plan[i, :] = new_plan[index, :]
                    self.fitness[i] = new_fit_best

            # 依据适应度重新排序
            self.position, self.plan = self.update(self.position, self.plan)
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
            # 放回
            self.plan[self.num_pop-1, :] = plan_iter
            # 保存每次迭代最优个体
            self.plan_one[n, :] = plan_iter

            self.t = self.t + 1
            # 迭代性能计算
            p1, p2 = self.cal_iter(plan_iter)
            if self.pointer1:
                print('剩余基地价值比例为：{}'.format(p1))
                print('预计打击目标比例为：{}'.format(p2))
            # 保存数据到csv文件

            if self.pointer2:

                with open('data_SPO_0.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([p1, p2])



    def run(self):
        # 初始化群体位置
        self.rand_init_pos()
        # 产生对应方案
        for a in range(self.num_pop):
            position = self.position[a, :]
            plan = self.transformation(position)
            self.plan[a, :] = plan
        # 进行升序排序，最小的在前面
        self.position, self.plan = self.update(self.position, self.plan)
        # 算法群体更新
        self.search()
        # 选择最优的个体
        best_plan = self.plan_one[self.iter_max-1, :].astype('int32').tolist()
        # print("输出方案为{}".format(best_plan))
        # print("可行性矩阵为{}".format(self.feasibility))
        return best_plan