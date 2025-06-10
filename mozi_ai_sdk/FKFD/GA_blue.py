import numpy as np
import random

class WTA_GA:
    def __init__(self, pij, wj, weapon_num, pop_size=50, generations=200, mutation_rate=0.1):
        """
        pij: [n_units x n_targets] 矩阵，单位对目标的毁伤概率
        wj: [n_targets] 向量，目标的价值
        weapon_num: [n_units] 向量，每个单位最多可使用的武器数量
        pop_size: 种群规模
        generations: 最大迭代次数
        mutation_rate: 变异概率
        """
        self.pij = np.array(pij)
        self.wj = np.array(wj)
        self.weapon_num = np.array(weapon_num)
        self.n_units, self.n_targets = self.pij.shape
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        """
        每个个体为一个 [n_units x n_targets] 的矩阵，每行一个非零元素
        """
        population = []
        for _ in range(self.pop_size):
            x = np.zeros((self.n_units, self.n_targets), dtype=int)
            for i in range(self.n_units):
                j = random.randint(0, self.n_targets - 1)
                x[i, j] = random.randint(1, self.weapon_num[i])
            population.append(x)
        return population

    def fitness(self, x):
        """
        计算适应度：目标毁伤期望总和
        f = sum_j wj[j] * (1 - prod_i (1 - pij[i][j]) ** x[i][j])
        """
        total = 0.0
        for j in range(self.n_targets):
            prob_survive = 1.0
            for i in range(self.n_units):
                prob_survive *= (1 - self.pij[i][j]) ** x[i][j]
            total += self.wj[j] * (1 - prob_survive)
        return total

    def select_parents(self, population, fitnesses):
        """
        轮盘赌选择
        """
        total_fit = sum(fitnesses)
        probs = [f / total_fit for f in fitnesses]
        selected = np.random.choice(range(self.pop_size), size=2, p=probs)
        return population[selected[0]], population[selected[1]]

    def crossover(self, parent1, parent2):
        """
        单点交叉：按单位维度（行）切分
        """
        point = random.randint(1, self.n_units - 1)
        child = np.vstack([parent1[:point], parent2[point:]])
        return child

    def mutate(self, individual):
        """
        变异操作：每个单位有概率改变目标或武器数量
        """
        for i in range(self.n_units):
            if random.random() < self.mutation_rate:
                j_old = np.argmax(individual[i])
                j_new = random.randint(0, self.n_targets - 1)
                weapon = random.randint(1, self.weapon_num[i])
                individual[i] = np.zeros(self.n_targets, dtype=int)
                individual[i][j_new] = weapon
        return individual

    def evolve(self):
        """
        主进化流程
        """
        population = self.initialize_population()
        best_individual = None
        best_fitness = -np.inf

        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]

            # 记录最优个体
            max_f = max(fitnesses)
            if max_f > best_fitness:
                best_fitness = max_f
                best_individual = population[np.argmax(fitnesses)]

            new_population = []
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_parents(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        return best_individual, best_fitness
