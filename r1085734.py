import Reporter
import numpy as np
import random

# Modify the class name to match your student number.
class r1085734:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.population_size = 200
        self.num_generations = 1000
        self.k_tournament = 5
        self.mutation_prob_swap = 0.05
        self.mutation_prob_insert = 0.1
        self.include_greedy = False

    def initialize_population(self, distanceMatrix):
        n = distanceMatrix.shape[0]
        population = []

        def is_valid_tour(tour):
            for i in range(len(tour) - 1):
                if np.isinf(distanceMatrix[tour[i], tour[i+1]]):
                    return False
            if np.isinf(distanceMatrix[tour[-1], tour[0]]):
                return False
            return True

        while len(population) < self.population_size - int(self.include_greedy):
            tour = [0] + random.sample(range(1, n), n - 1)
            if is_valid_tour(tour):
                population.append(np.array(tour))

        if self.include_greedy:
            greedy_tour = self.greedy_solution(distanceMatrix)
            population.append(greedy_tour)

        return population

    def greedy_solution(self, distanceMatrix):
        n = distanceMatrix.shape[0]
        unvisited = set(range(1, n))
        tour = [0]
        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, key=lambda x: distanceMatrix[last][x] if not np.isinf(distanceMatrix[last][x]) else float('inf'))
            if np.isinf(distanceMatrix[last][next_city]):
                break
            tour.append(next_city)
            unvisited.remove(next_city)
        return np.array(tour)

    def evaluate_fitness(self, tour, distanceMatrix):
        total_distance = 0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            if np.isinf(distanceMatrix[from_city][to_city]):
                return float('inf')
            total_distance += distanceMatrix[from_city][to_city]
        return total_distance

    def k_tournament_selection(self, population, fitnesses):
        selected = random.sample(list(zip(population, fitnesses)), self.k_tournament)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def swap_mutation(self, tour):
        if random.random() < self.mutation_prob_swap:
            i, j = random.sample(range(1, len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    def insert_mutation(self, tour):
        if random.random() < self.mutation_prob_insert:
            i, j = sorted(random.sample(range(1, len(tour)), 2))
            city = tour.pop(j)
            tour.insert(i, city)
        return tour

    def pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        p1, p2 = sorted(random.sample(range(1, size), 2))
        child = [-1] * size
        child[0] = 0
        child[p1:p2] = parent1[p1:p2]

        for i in range(p1, p2):
            if parent2[i] not in child:
                val = parent2[i]
                pos = i
                while True:
                    val = parent1[pos]
                    pos = parent2.tolist().index(val)
                    if child[pos] == -1:
                        child[pos] = parent2[i]
                        break
        for i in range(1, size):
            if child[i] == -1:
                child[i] = parent2[i]
        return np.array(child)

    def edge_crossover(self, parent1, parent2):
        size = len(parent1)
        edge_map = {i: set() for i in parent1}
        for p in [parent1, parent2]:
            for i in range(size):
                edge_map[p[i]].add(p[(i - 1) % size])
                edge_map[p[i]].add(p[(i + 1) % size])

        current = parent1[0]
        child = [current]
        while len(child) < size:
            for edges in edge_map.values():
                edges.discard(current)
            if edge_map[current]:
                next_city = min(edge_map[current], key=lambda x: len(edge_map[x]))
            else:
                remaining = [c for c in parent1 if c not in child]
                next_city = random.choice(remaining)
            child.append(next_city)
            current = next_city
        return np.array(child)

    def optimize(self, filename):
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        population = self.initialize_population(distanceMatrix)
        fitnesses = [self.evaluate_fitness(tour, distanceMatrix) for tour in population]

        bestObjective = float('inf')
        bestSolution = None

        for generation in range(self.num_generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.k_tournament_selection(population, fitnesses)
                parent2 = self.k_tournament_selection(population, fitnesses)
                if random.random() < 0.5:
                    child = self.pmx_crossover(parent1.copy(), parent2.copy())
                else:
                    child = self.edge_crossover(parent1.copy(), parent2.copy())
                child = self.swap_mutation(child.tolist())
                child = self.insert_mutation(child)
                new_population.append(np.array(child))

            combined_population = population + new_population
            combined_fitnesses = [self.evaluate_fitness(tour, distanceMatrix) for tour in combined_population]
            sorted_combined = sorted(zip(combined_population, combined_fitnesses), key=lambda x: x[1])
            population = [x[0] for x in sorted_combined[:self.population_size]]
            fitnesses = [x[1] for x in sorted_combined[:self.population_size]]

            meanObjective = np.mean(fitnesses)
            bestObjective = fitnesses[0]
            bestSolution = population[0]

            print(f"Generation {generation+1}: Mean Fitness = {meanObjective:.2f}, Best Fitness = {bestObjective:.2f}")

            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        return 0

if __name__ == "__main__":
    solver = r1085734()
    solver.optimize("tour250.csv")