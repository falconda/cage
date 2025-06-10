import random
import numpy as np
import copy
import csv

class DWHO_WTA(object):
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
        self.num_pop = 30  # 群体规模
        self.iter_max = 100
        self.PS = 0.2
        self.PC = 0.13
        self.alpha = 0.5    # 用于目标函数加权
        # NStallion 种马数量 NFoal 小马数量
        self.MIN = min(num_weapon, num_target)   # 注意分三种情况讨论
        num_stallion = int(self.PS*self.num_pop)
        self.NStallion = np.zeros((num_stallion, self.MIN))
        self.NFoal = np.zeros((self.num_pop - num_stallion, self.MIN))
        self.plan_stallion = np.full((self.NStallion.shape[0], self.num_weapon), -1)
        self.plan_foal = np.full((self.NFoal.shape[0], self.num_weapon), -1)
        self.plan_one = np.full((self.iter_max, self.num_weapon), -1)
        # 适应度保存
        self.fitness_stallion = np.zeros(self.NStallion.shape[0])  # [1, self.num_pop]
        self.fitness_foal = np.zeros(self.NFoal.shape[0])
        self.fitness_save = np.zeros((self.iter_max + 1, self.num_pop))  # [self.iter_max+1, self.num_pop]

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

    # 更新步骤,全部种马群体or全部小马群体,并进行排序？选择适应度最高的
    def update(self, pos_pop, plan_pop):
        length = plan_pop.shape[0]
        fitness = np.zeros(length)
        for n in range(length):
            single_plan = plan_pop[n, :]
            fit = self.cal_single_fitness(single_plan)
            fitness[n] = fit

        # 排序？还是只选出最大的->只选最大的
        fitness_sort = sorted(fitness, reverse=True)
        fitness_list = fitness.tolist()
        sort_indices = []
        # 按照适应度从大到小的顺序，搜索原来在fitness的位置，保存在sort_indices中
        for index, value in enumerate(fitness_sort):
            sort_indices.append(fitness_list.index(value))
        # 更新种马个体
        pos_stallion = pos_pop[sort_indices[0], :]
        plan_stallion = plan_pop[sort_indices[0], :]
        fit_stallion = fitness[sort_indices[0]]
        # 更新小马群体, 降序排序
        pos_foal = np.zeros((length-1, self.MIN))
        plan_foal = np.full((length-1, self.num_weapon), -1)
        fit_foal = np.zeros(length-1)
        for m in range(1, length):  # 去掉最优的，按顺序排入
            pos_foal[m-1, :] = pos_pop[sort_indices[m]]
            plan_foal[m-1, :] = plan_pop[sort_indices[m]]
            fit_foal[m-1] = fitness[sort_indices[m]]

        return pos_stallion, pos_foal, plan_stallion, plan_foal, fit_stallion, fit_foal

    def rand_init_pos(self):
        # 初始化
        for j in range(self.NStallion.shape[0]):
            self.NStallion[j, :] = self.random_pos()
        for i in range(self.NFoal.shape[0]):
            self.NFoal[i, :] = self.random_pos()
            # print('初始化第{}个方案为{}'.format(i, self.plan[i, :]))

    # 产生实数向量，内容是0-1的向量
    def random_pos(self):
        pos = np.random.rand(self.MIN)
        return pos

    # 计算每个群体的参数
    def cal_factor(self, m):
        TDR = 1 - self.t / self.iter_max
        stallion_pos = self.NStallion[m, :]
        z = np.random.uniform(0, TDR, size=self.MIN)
        r1 = random.random()
        r2 = np.random.rand(self.MIN)
        idx = np.zeros(self.MIN)
        for j in range(len(z)):
            if z[j] == 0:
                idx[j] = 1
            else:
                idx[j] = 0
        idx_bool = idx.astype(np.bool)
        r3 = r1 * idx_bool + r2 * (~idx_bool)
        rr = -2 + 4 * r3
        return rr, r3, stallion_pos

    #  小马的放牧操作
    def grazing(self, rr, r3, pos_stallion, pos_foal):
        pos_foal_done = 2 * r3 * np.cos(2*np.pi*rr) * (pos_stallion - pos_foal) + pos_stallion

        return pos_foal_done

    # 小马的交配操作
    def crossover(self, count, Ngroup):
        arr = np.arange(self.NStallion.shape[0])        # 产生种马的整数向量
        arr1 = np.delete(arr, np.where(arr == count))   # 删除选中组的序号
        select_indices = np.random.choice(arr1, size=2, replace=False)  # 随机选择组
        select_pos0 = self.NFoal[select_indices[0]*Ngroup + Ngroup-1, :]   # 选中组最后一个个体
        select_pos1 = self.NFoal[select_indices[1]*Ngroup + Ngroup-1, :]   # 选中组最后一个个体
        pos_crossover = (select_pos0 + select_pos1)/2

        return pos_crossover

    # 种马迁徙操作
    def migration(self, R, r3, rr, pos_stallion, pos_WH):
        # WH_pos 选择最优的种马？整个种马和小马群体的个体值？
        if R < 0.5:
            pos_migration = 2 * r3 * np.cos(2 * np.pi * rr) * (pos_WH - pos_stallion) + pos_WH
        else:
            pos_migration = 2 * r3 * np.cos(2 * np.pi * rr) * (pos_WH - pos_stallion) - pos_WH
        return pos_migration

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
        rand = random.randint(0, 2)
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
        # 确定一个种群里的个数
        Ngroup = int(self.NFoal.shape[0] / self.NStallion.shape[0])
        for i in range(self.iter_max):
            # 每一个群体计算一次：群体数量：NStallion.shape[0] 群体里小马数量：Ngroup
            indice = np.argmax(self.fitness_stallion)
            pos_WH = self.NStallion[indice, :]

            for m in range(self.NStallion.shape[0]):
                # 初始化相关参数，用于位置更新
                rr, r3, pos_stallion = self.cal_factor(m)
                # 小马个体更新位置
                for n in range(Ngroup):
                    pos_foal = self.NFoal[m * Ngroup + n, :]
                    rand = random.random()
                    if rand > self.PC:
                         pos_gra = self.grazing(rr, r3, pos_stallion, pos_foal)
                         self.NFoal[m*Ngroup + n, :] = pos_gra
                    else:
                        pos_cos = self.crossover(m, Ngroup)
                        self.NFoal[m*Ngroup + n, :] = pos_cos
                    # 离散化并计算适应度,保存
                    plan = self.transformation(self.NFoal[m * Ngroup + n, :])
                    self.plan_foal[m * Ngroup + n, :] = plan
                    self.fitness_foal[m*Ngroup+n] = self.cal_single_fitness(plan)
                # 单个种群的种马位置更新
                R = random.random()
                pos_stallion_mig = self.migration(R, r3, rr, pos_WH, pos_stallion)
                self.NStallion[m, :] = pos_stallion_mig
                # 离散化种马个体并计算适应度
                plan_stallion = self.transformation(pos_stallion_mig)
                self.plan_stallion[m, :] = plan_stallion
                self.fitness_stallion[m] = self.cal_single_fitness(plan_stallion)
                # 重新选择当前组的种马：按照适应度排序
                pos_group = self.NFoal[m * Ngroup:(m+1) * Ngroup, :]
                pos_pop = np.row_stack((pos_stallion, pos_group))
                plan_group = self.plan_foal[m * Ngroup:(m+1) * Ngroup, :]
                plan_pop = np.row_stack((plan_stallion, plan_group))
                pos_stallion, pos_foal, plan_stallion, plan_foal, fit_stallion, fit_foal = self.update(pos_pop, plan_pop)
                # 更新，计算结果更新到种马个体和小马群体去
                self.NStallion[m, :] = pos_stallion
                self.plan_stallion[m, :] = plan_stallion
                self.fitness_stallion[m] = fit_stallion
                # 小马群体
                self.NFoal[m * Ngroup:(m+1) * Ngroup, :] = pos_foal
                self.plan_foal[m * Ngroup:(m+1) * Ngroup, :] = plan_foal
                self.fitness_foal[m * Ngroup:(m+1) * Ngroup] = fit_foal
            self.t = self.t + 1
            # 更新完群体，对最优解进行邻域搜索
            # 选择本次种马个体中最优个体 中间方案，未进行局部搜索
            plan_mid = self.plan_stallion[indice, :]
            # plan_mid = self.transformation(pos_WH)
            # 进行局部搜索
            if self.num_weapon > 1:
                for j in range(15):
                    plan_search = self.local_search(plan_mid)
            else:
                plan_search = plan_mid
            # 历史迭代，选择相比于之前更优秀的
            if i == 0:
                plan_iter = plan_search
                self.plan_one[0, :] = plan_iter
            else:
                plan_pre = self.plan_one[i-1, :]
                fit_now = self.cal_single_fitness(plan_search)
                fit_pre = self.cal_single_fitness(plan_pre)
                if fit_now > fit_pre:
                    plan_iter = plan_search
                else:
                    plan_iter = plan_pre
            self.plan_stallion[indice, :] = plan_iter
            # 保存每次迭代最优个体
            self.plan_one[i, :] = plan_iter

            self.t = self.t + 1
            # 迭代性能计算
            p1, p2 = self.cal_iter(plan_iter)
            # print('剩余基地价值比例为：{}'.format(p1))
            # print('预计打击目标比例为：{}'.format(p2))
            # 保存数据到csv文件
            '''
            with open('data_DWHO_0.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([p1, p2])             
            '''

    def run(self):
        # 初始化种马和小马
        self.rand_init_pos()
        # 种马群体进行离散化，得到方案plan
        for k in range(self.NStallion.shape[0]):
            single_pos = self.NStallion[k, :]
            single_plan = self.transformation(single_pos)
            self.plan_stallion[k, :] = single_plan
            # 计算各个种马适应度
            self.fitness_stallion[k] = self.cal_single_fitness(single_plan)
        # 算法群体更新
        self.search()
        # 选择最优的种马个体
        best_plan = self.plan_one[self.iter_max-1, :].astype('int32').tolist()
        # print("输出方案为{}".format(best_plan))
        # print("可行性矩阵为{}".format(self.feasibility))
        return best_plan