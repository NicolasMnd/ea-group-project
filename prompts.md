# Prompts used for generating the code
Record of the prompts and postprocessing used for the current codebase.

## Initial prompt
Hi, I would like you to write for me and my team an implementation in Python and Numpy of an evolutionary algorithm to solve the traveling salesperson problem.

In this specific setting, the graph is directed and the weights are recorded in a weight matrix that has to be read from a .csv file. Weight are floating point values, with Inf in position (i, j) indicating a non existing edge between vertex i and vertex j.

It is mandatory that you conform to the .py template I provided in attach and do not edit any of the code there, just add new code. Also try to maintain the same style.

Specifically, you should:
- Use a permutation-based representation, where the cities are represented in a numpy array in the order they are visited
- Generate the initial population of tours completely randomly but keeping 1 as the first city of the tour. Also check, using the weight matrix, that the edges used in the tours you generated exist. Keep generating new initial tours until you find the necessary number of valid solutions to start with. Also insert in the initial population a solution to the problem found using a greedy heuristic for TSP. Control this last feature with a boolean to disable it.
- k-tournament selection should be used as a selection criterion to select the parents for the recombination step and an elimination operator to keep the population size constant. First merge the offspring with the previous population, then perform elimination. Do not just substitute the new offspring with the previous population.
- Implement swap and insert mutation operators, each with a configurable mutation probability
- Implement recombination with Partially Mapped Crossover (PMX) and edge crossover
- Implement a stopping criterion where the algorithm stops after a certain number of iterations

Some other instructions:
- All relevant parameters should be set in the __init__ method of the class
- Write everythign in a single python file
- Log the mean and best fitness of the population at every iteration

## Minor adjustment to add main and print statements
Also include a main method to run it on a file in the current folder called tour50.csv

Add print statements as explained to see intermediate results while the algorithm is running