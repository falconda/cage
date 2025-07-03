import numpy as np
import random

class WTA_GA:
    def __init__(self, pij, wj, weapon_num, tij, pop_size=50, generations=200, mutation_rate=0.1, elite_ratio=0.1):
        """
        pij: [n_units x n_targets] 单位对目标的毁伤概率
        wj: [n_targets] 每个目标的价值
        weapon_num: [n_units] 每个单位可用的最大武器数
        tij: [n_units x n_targets] 每个单位摧毁目标所需的武器数
        """
        self.pij = np.array(pij)
        self.wj = np.array(wj)
        self.weapon_num = np.array(weapon_num)
        self.tij = np.array(tij)
        self.n_units, self.n_targets = self.pij.shape
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio

    def initialize_population(self):
        """
        生成满足：
        - 每行最多一个1（单位攻击一个目标）
        - 每列最多一个1（目标被最多一个单位攻击）
        """
        population = []
        for _ in range(self.pop_size):
            x = np.zeros((self.n_units, self.n_targets), dtype=int)
            min_len = min(self.n_units, self.n_targets)
            perm = np.random.permutation(self.n_targets)
            if self.n_units <= self.n_targets:
                for i in range(self.n_units):
                    x[i, perm[i]] = 1
            else:
                unit_indices = np.random.permutation(self.n_units)[:self.n_targets]
                for i, unit_idx in enumerate(unit_indices):
                    x[unit_idx, perm[i]] = 1
            population.append(x)
        return population

    def get_actual_assignment(self, x):
        """
        根据 x (0-1 配对矩阵)，生成实际武器分配矩阵：min(tij, weapon_num[i])
        """
        x_actual = np.zeros_like(x, dtype=float)
        for i in range(self.n_units):
            for j in range(self.n_targets):
                if x[i][j] == 1:
                    x_actual[i][j] = min(self.tij[i][j], self.weapon_num[i])
        return x_actual

    def fitness(self, x):
        """
        计算适应度：每个单位与目标配对后，根据 min(tij, weapon_num[i]) 计算真实分配
        然后按比例毁伤价值
        """
        x_actual = self.get_actual_assignment(x)
        total = 0.0
        for j in range(self.n_targets):
            destroy_ratio = 0.0
            for i in range(self.n_units):
                if self.tij[i][j] > 0:
                    destroy_ratio += self.pij[i][j] * x_actual[i][j] / self.tij[i][j]
            total += self.wj[j] * min(1.0, destroy_ratio)
        return total

    def select_parents(self, population, fitnesses):
        total_fit = sum(fitnesses)
        probs = [f / total_fit for f in fitnesses]
        selected = np.random.choice(range(self.pop_size), size=2, p=probs)
        return population[selected[0]], population[selected[1]]

    def crossover(self, parent1, parent2):
        if self.n_units > 1:
            point = random.randint(1, self.n_units - 1)
        else:
            return parent1.copy()
        child = np.vstack([parent1[:point], parent2[point:]])
        return child

    def mutate(self, individual):
        """
        每行有概率变异为另一个目标
        变异时不考虑 tij 和 weapon_num 限制
        """
        for i in range(self.n_units):
            if random.random() < self.mutation_rate:
                j_new = random.randint(0, self.n_targets - 1)
                individual[i] = np.zeros(self.n_targets, dtype=int)
                individual[i][j_new] = 1
        return individual

    def evolve(self):
        population = self.initialize_population()
        best_individual = None
        best_fitness = -np.inf

        fitness_history = []

        elite_num = max(1, int(self.pop_size * self.elite_ratio))

        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            fitness_history.append(fitnesses)

            # 更新最优解
            max_f = max(fitnesses)
            if max_f > best_fitness:
                best_fitness = max_f
                best_individual = population[np.argmax(fitnesses)]

            # --- 精英保留 ---
            # 获取当前代中适应度最高的 elite_num 个体
            elite_indices = np.argsort(fitnesses)[-elite_num:]
            elites = [population[i] for i in elite_indices]

            # --- 其余通过进化产生 ---
            new_population = elites.copy()
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_parents(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        return best_individual, best_fitness, fitness_history


