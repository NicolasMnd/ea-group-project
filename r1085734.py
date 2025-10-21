import Reporter
import numpy as np

class r1085734:

    def __init__(self, seed=None):
        self.k = 70
        self.rng = np.random.default_rng(seed)
        self.reporter = Reporter.Reporter(self.__class__.__name__)
    
    def is_valid_solution(self, solution, weight_matrix):
        """
        Checks if a solution is valid (no edge has infinite weight).
        """
        for i in range(self.n):
            from_node = solution[i]
            # wrap around to form a cycle
            to_node = solution[(i + 1) % self.n]  
            if weight_matrix[from_node, to_node] == np.inf:
                return False
        return True
    

    def generate_initial_population(self, population_size):
        """
        Generates a population of valid TSP solutions.
        
        Each solution is a permutation of the node indices.
        Invalid solutions (with np.inf edges) are discarded.
        
        Parameters:
        - population_size (int): Number of solutions to generate.
        
        Returns:
        - population (np.ndarray): Array of shape (population_size, n) 
        with each row a valid tour.
        """
        population = []
        attempts = 0
        max_attempts = population_size * 10  # prevent infinite loops

        while len(population) < population_size and attempts < max_attempts:
            candidate = self.rng.permutation(self.n)
            if self.is_valid_solution(candidate, self.distanceMatrix):
                population.append(candidate)
            attempts += 1

        if len(population) < population_size:
            raise ValueError("Could not generate enough valid solutions. Check the weight matrix.")

        return np.array(population)
    
    def fitness(self, solution):
        """
        Computes the total length of the tour represented by the solution.
        
        Parameters:
        - solution (np.ndarray): A permutation of node indices representing a tour.
        
        Returns:
        - total_distance (float): Sum of weights along the tour. Returns np.inf if any edge is invalid.
        """
        total_distance = 0.0
        for i in range(self.n):
            from_node = solution[i]
            to_node = solution[(i + 1) % self.n]  # wrap around to form a cycle
            weight = self.distanceMatrix[from_node, to_node]
            if weight == np.inf:
                return np.inf
            total_distance += weight
        return total_distance
    

    def evaluate_population(self, population):
        """
        Evaluates a population of solutions and returns:
        - mean objective value
        - best objective value
        - best solution

        Parameters:
        - population (np.ndarray): Array of shape (population_size, n)

        Returns:
        - meanObjective (float)
        - bestObjective (float)
        - bestSolution (np.ndarray)
        """
        fitness_values = np.array([self.fitness(ind) for ind in population])
        meanObjective = np.mean(fitness_values)
        best_idx = np.argmin(fitness_values)
        bestObjective = fitness_values[best_idx]
        bestSolution = population[best_idx]
        return meanObjective, bestObjective, bestSolution
    
    def tournament_selection(self, k, num_selected, population):
        """
        Selects individuals using k-tournament selection.

        Parameters:
        - k (int): Number of individuals in each tournament.
        - num_selected (int): Number of individuals to select.
        - population (np.ndarray): Current population.

        Returns:
        - selected (np.ndarray): Selected individuals for recombination.
        """
        selected = []
        for _ in range(num_selected):
            tournament = self.rng.choice(population, size=k, replace=False)
            fitnesses = np.array([self.fitness(ind) for ind in tournament])
            winner = tournament[np.argmin(fitnesses)]
            selected.append(winner)
        return np.array(selected)
    

    def recombine(self, parents):
        """
        Performs ordered crossover (OX) on pairs of parents.

        Parameters:
        - parents (np.ndarray): Array of selected parents.

        Returns:
        - offspring (np.ndarray): Array of recombined children.
        """
        offspring = []
        num_parents = len(parents)
        for i in range(0, num_parents, 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % num_parents]  # wrap around if odd number

            # Ordered crossover
            start, end = sorted(self.rng.choice(self.n, size=2, replace=False))
            child = [-1] * self.n
            child[start:end] = parent1[start:end]

            # Fill remaining positions from parent2
            p2_idx = 0
            for j in range(self.n):
                if child[j] == -1:
                    while parent2[p2_idx] in child:
                        p2_idx += 1
                    child[j] = parent2[p2_idx]

            offspring.append(np.array(child))
        return np.array(offspring)
    
    def mutate(self, population, mutation_rate=0.2):
        """
        Mutates a percentage of the population by swapping two cities in each selected individual.

        Parameters:
        - population (np.ndarray): Current population.
        - mutation_rate (float): Fraction of individuals to mutate (e.g., 0.2 = 20%).

        Returns:
        - mutated_population (np.ndarray): Population after mutation.
        """
        mutated_population = population.copy()
        num_to_mutate = int(len(population) * mutation_rate)

        indices = self.rng.choice(len(population), size=num_to_mutate, replace=False)
        for idx in indices:
            individual = mutated_population[idx].copy()
            i, j = self.rng.choice(self.n, size=2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
            mutated_population[idx] = individual

        return mutated_population


    def optimize(self, filename):
        with open(filename) as file:
            self.distanceMatrix = np.loadtxt(file, delimiter=",")
            self.n = self.distanceMatrix.shape[0]

        population_size = 500
        population = self.generate_initial_population(population_size)

        i = 1
        while i <= 10000:
            meanObjective, bestObjective, bestSolution = self.evaluate_population(population)

            print(f"\n---Iteration {i}---")
            print(f"Mean Objective: {meanObjective}")
            print(f"Best Objective: {bestObjective}")
            print(f"Best Solution: {bestSolution}")
            print("----------------------")

            # Selection
            selected_parents = self.tournament_selection(k=self.k, num_selected=population_size, population=population)

            # Recombination
            offspring = self.recombine(selected_parents)

            # Mutation
            offspring = self.mutate(offspring, mutation_rate=0.3)

            # Combine and eliminate
            combined_population = np.vstack((population, offspring))
            fitness_values = np.array([self.fitness(ind) for ind in combined_population])
            best_indices = np.argsort(fitness_values)[:population_size]
            population = combined_population[best_indices]

            # Report
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

            i += 1

        return 0
    
def main():
    solver = r1085734(seed=42)
    solver.optimize("tour50.csv")

if __name__ == "__main__":
    main()