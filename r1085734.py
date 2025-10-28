import numpy as np
import Reporter

TOUR_FILE = "tour50.csv"

class r1085734:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.population_size = 200
        self.num_iterations = 500
        self.k_tournament = 5
        self.mutation_prob_swap = 0.05
        self.mutation_prob_insert = 0.1
        self.include_greedy = False

    def initialize_population(self, distance_matrix):
        n = distance_matrix.shape[0]
        population = []

        def is_valid_tour(tour):
            for i in range(len(tour) - 1):
                if np.isinf(distance_matrix[tour[i], tour[i+1]]):
                    return False
            if np.isinf(distance_matrix[tour[-1], tour[0]]):
                return False
            return True

        while len(population) < self.population_size - int(self.include_greedy):
            tour = np.concatenate(([0], np.random.permutation(np.arange(1, n))))
            if is_valid_tour(tour):
                population.append(tour)

        if self.include_greedy:
            greedy_tour = self.greedy_solution(distance_matrix)
            population.append(greedy_tour)

        return population

    def greedy_solution(self, distance_matrix):
        n = distance_matrix.shape[0]
        unvisited = set(range(1, n))
        tour = [0]
        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, key=lambda x: distance_matrix[last][x] if not np.isinf(distance_matrix[last][x]) else float('inf'))
            if np.isinf(distance_matrix[last][next_city]):
                break
            tour.append(next_city)
            unvisited.remove(next_city)
        return np.array(tour)

    def evaluate_fitness(self, tour, distance_matrix):
        total_distance = 0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            if np.isinf(distance_matrix[from_city][to_city]):
                return float('inf')
            total_distance += distance_matrix[from_city][to_city]
        return total_distance

    def k_tournament_selection(self, population, fitnesses):
        indices = np.random.choice(len(population), self.k_tournament, replace=False)
        selected = [(population[i], fitnesses[i]) for i in indices]
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def swap_mutation(self, tour):
        if np.random.rand() < self.mutation_prob_swap:
            i, j = np.random.choice(np.arange(1, len(tour)), 2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    def insert_mutation(self, tour):
        if np.random.rand() < self.mutation_prob_insert:
            i, j = np.sort(np.random.choice(np.arange(1, len(tour)), 2, replace=False))
            city = tour[j]
            tour = np.delete(tour, j)
            tour = np.insert(tour, i, city)
        return tour

    def pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        p1, p2 = np.sort(np.random.choice(np.arange(1, size), 2, replace=False))
        child = -np.ones(size, dtype=int)
        child[0] = 0
        child[p1:p2] = parent1[p1:p2]

        for i in range(p1, p2):
            if parent2[i] not in child:
                val = parent2[i]
                pos = i
                while True:
                    val = parent1[pos]
                    pos = np.where(parent2 == val)[0][0]
                    if child[pos] == -1:
                        child[pos] = parent2[i]
                        break
        for i in range(1, size):
            if child[i] == -1:
                child[i] = parent2[i]
        return child

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
                next_city = np.random.choice(remaining)
            child.append(next_city)
            current = next_city
        return np.array(child)

    def optimize(self, filename):
        with open(filename) as file:
            distance_matrix = np.loadtxt(file, delimiter=",")

        population = self.initialize_population(distance_matrix)
        fitnesses = [self.evaluate_fitness(tour, distance_matrix) for tour in population]

        best_objective = float('inf')
        best_solution = None

        for iteration in range(self.num_iterations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.k_tournament_selection(population, fitnesses)
                parent2 = self.k_tournament_selection(population, fitnesses)
                if np.random.rand() < 0.5:
                    child = self.pmx_crossover(parent1.copy(), parent2.copy())
                else:
                    child = self.edge_crossover(parent1.copy(), parent2.copy())
                child = self.swap_mutation(child)
                child = self.insert_mutation(child)
                new_population.append(child)

            combined_population = population + new_population
            combined_fitnesses = [self.evaluate_fitness(tour, distance_matrix) for tour in combined_population]
            sorted_combined = sorted(zip(combined_population, combined_fitnesses), key=lambda x: x[1])
            population = [x[0] for x in sorted_combined[:self.population_size]]
            fitnesses = [x[1] for x in sorted_combined[:self.population_size]]

            mean_objective = np.mean(fitnesses)
            best_objective = fitnesses[0]
            best_solution = population[0]

            print(f"----iteration {iteration+1}----")
            print(f"mean fitness={mean_objective:.2f}")
            print(f"best fitness={best_objective:.2f}")
            print("----------------------")

            timeLeft = self.reporter.report(mean_objective, best_objective, best_solution)
            if timeLeft < 0:
                break

        return 0

if __name__ == "__main__":
    solver = r1085734()
    solver.optimize(TOUR_FILE)
    