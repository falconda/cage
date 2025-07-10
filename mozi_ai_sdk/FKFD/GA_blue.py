import numpy as np
import random

class WTA_GA:
    def __init__(self, pij, wj, weapon_num, tij, pop_size=50, generations=400, mutation_rate=0.1, elite_ratio=0.1):
        self.pij = np.array(pij)
        self.wj = np.array(wj)
        self.weapon_num = np.array(weapon_num)
        self.tij = np.array(tij)
        self.n_units, self.n_targets = self.pij.shape
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio

        if self.n_units > self.n_targets:
            self.pos_dim = self.n_units  # 控制“谁出战+打谁”
        else:
            self.pos_dim = self.n_targets  # 控制“目标选择优先级”

    def position_to_plan(self, position: np.ndarray) -> np.ndarray:
        plan = -1 * np.ones(self.n_units, dtype=int)

        if self.n_units > self.n_targets:
            # 情况1：单位多于目标
            # 取 position 中最小的 n_targets 个单位编号
            top_unit = np.argsort(position)
            top_units = top_unit[:self.n_targets] # 例如 [0, 4, 2]
            for target_id, unit_id in enumerate(top_units):
                plan[unit_id] = target_id  # 给这些单位分配目标 0, 1, 2,...
        else:
            # 情况2：目标多于单位
            top_targets = np.argsort(position)[:self.n_units]  # 例如 [3, 1, 0]
            plan = top_targets  # 每个单位分配一个唯一目标
        return plan

    def plan_to_matrix(self, plan: np.ndarray) -> np.ndarray:
        """根据 plan 映射出单位-目标 0/1 矩阵"""
        x = np.zeros((self.n_units, self.n_targets), dtype=int)
        used_targets = set()
        for i in range(self.n_units):
            j = plan[i]
            if j == -1 or j in used_targets:
                continue
            x[i][j] = 1
            used_targets.add(j)
        return x

    def get_actual_assignment(self, x: np.ndarray) -> np.ndarray:
        """根据 0/1 配对矩阵生成实际武器分配矩阵"""
        x_actual = np.zeros_like(x, dtype=float)
        for i in range(self.n_units):
            for j in range(self.n_targets):
                if x[i][j] == 1:
                    x_actual[i][j] = min(self.tij[i][j], self.weapon_num[i])
        return x_actual

    def fitness(self, individual: dict) -> float:
        """适应度函数：根据个体的 plan 生成分配矩阵并计算目标毁伤价值"""
        plan = individual["plan"]
        x = self.plan_to_matrix(plan)
        x_actual = self.get_actual_assignment(x)
        total = 0.0
        for j in range(self.n_targets):
            destroy_ratio = 0.0
            for i in range(self.n_units):
                if self.tij[i][j] > 0:
                    destroy_ratio += self.pij[i][j] * x_actual[i][j] / self.tij[i][j]
            total += self.wj[j] * min(1.0, destroy_ratio)
        return total

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            position = np.random.rand(self.pos_dim)
            plan = self.position_to_plan(position)
            individual = {"position": position, "plan": plan}
            population.append(individual)
        return population

    def select_parents(self, population, fitnesses):
        total_fit = sum(fitnesses)
        probs = [f / total_fit for f in fitnesses]
        selected = np.random.choice(range(self.pop_size), size=2, p=probs)
        return population[selected[0]], population[selected[1]]

    def crossover(self, parent1: dict, parent2: dict) -> dict:
        alpha = random.random()
        new_pos = alpha * parent1["position"] + (1 - alpha) * parent2["position"]
        new_plan = self.position_to_plan(new_pos)
        return {"position": new_pos, "plan": new_plan}

    def mutate(self, individual: dict) -> dict:
        pos = individual["position"].copy()
        mutation_mask = np.random.rand(self.pos_dim) < self.mutation_rate
        noise = np.random.normal(0, 0.1, size=self.pos_dim)
        pos = pos + mutation_mask * noise
        new_plan = self.position_to_plan(pos)
        return {"position": pos, "plan": new_plan}

    def evolve(self):
        population = self.initialize_population()
        best_individual = None
        best_fitness = -np.inf
        fitness_history = []

        elite_num = max(1, int(self.pop_size * self.elite_ratio))

        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]
            fitness_history.append(fitnesses)

            # 记录当前最优
            max_f = max(fitnesses)
            if max_f > best_fitness:
                best_fitness = max_f
                best_individual = population[np.argmax(fitnesses)]

            # 精英保留
            elite_indices = np.argsort(fitnesses)[-elite_num:]
            elites = [population[i] for i in elite_indices]

            # 生成下一代
            new_population = elites.copy()
            while len(new_population) < self.pop_size:
                p1, p2 = self.select_parents(population, fitnesses)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        best_matrix = self.plan_to_matrix(best_individual["plan"])
        return best_matrix, best_individual["plan"], best_fitness, fitness_history
