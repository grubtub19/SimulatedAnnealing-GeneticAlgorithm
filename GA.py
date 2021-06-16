import numpy as np
import math
import random
import ga_eval
import copy
from FunctionParams import FunctionParamsGA
from Grapher import Grapher

class GA:

    def __init__(self):
        self.population_size = 200
        self.mutation_prob = 0.2
        self.elite_clones = 1
        self.num_dimensions = 2
        self.max_steps = 1000
        self.normalize_fitness_low = 40
        self.normalize_fitness_high = 100
        self.population = list()
        self.elite_indices = list()
        for index in range(self.elite_clones):
            self.elite_indices.append(-1)
        self.function_list = [FunctionParamsGA('bump', 0, 5, ga_eval.bump, ga_eval.bump_c),
                              FunctionParamsGA('sphere', -5, 5, ga_eval.sphere, ga_eval.sphere_c),
                              FunctionParamsGA('griew', 0, 200, ga_eval.griew, ga_eval.griew_c),
                              FunctionParamsGA('shekel', 0, 10, ga_eval.shekel, ga_eval.shekel_c),
                              FunctionParamsGA('micha', -100, 100, ga_eval.micha, ga_eval.micha_c),
                              FunctionParamsGA('langermann', 0, 10, ga_eval.langermann, ga_eval.langermann_c),
                              FunctionParamsGA('odd_square', -5 * math.pi, 5 * math.pi, ga_eval.odd_square, ga_eval.odd_square_c)
                              ]

        self.lowest_raw_fitness = float("inf")
        self.highest_raw_fitness = float("-inf")
        self.average_raw_fitness = 0
        self.best_chromosome = None

    class Chromosome:
        def __init__(self, num_dimensions: int, lower_limit: int, upper_limit: int):
            self.lower_limit = lower_limit
            self.upper_limit = upper_limit
            self.accuracy = 16
            self.num_dimensions = num_dimensions
            self.raw_fitness = 0
            self.fitness = 0
            self.bit_string = self.init_random()

        def init_random(self):
            result = np.zeros(self.num_dimensions * self.accuracy)
            for index in range(len(result)):
                result[index] = random.randrange(2)
            return result

        def chromosome_to_float(self):
            lower = 0
            upper = self.accuracy
            float_array = np.empty(0)

            while upper <= (self.accuracy * self.num_dimensions):
                #print("upper: " + str(upper) + ", lower: " + str(lower))
                bit_array = self.bit_string[lower:upper]
                #print("len(bit_array): " + str(len(bit_array)))
                float_array = np.append(float_array, self.bit_to_bounds(bit_array))
                lower += self.accuracy
                upper += self.accuracy
            #print("values: " + str(float_array))
            return float_array

        def bit_to_bounds(self, bit_array):
            value = self.bit_to_float(bit_array)
            result = (value / (math.pow(2, self.accuracy) - 1)) * (self.upper_limit - self.lower_limit) + self.lower_limit
            #print("float: " + str(result))
            return result

        def bit_to_float(self, bit_array):
            length = len(bit_array)
            result = 0
            for index, value in enumerate(bit_array):
                if value == 1:
                    result += math.pow(2, length - index - 1)
                    # print("result += " + str(math.pow(2, length - index)))
            #print("bit: " + str(result))
            return result

        # Gets raw fitness value of Chromosome
        def eval_fitness(self, fitness_fn: callable):
            self.raw_fitness = fitness_fn(self.chromosome_to_float())

        def crossover(self, chromosome):
            # New chromosome is copy of one parent
            new_chromosome = copy.deepcopy(self)
            # Determine the crossover point
            crossover_point = random.randrange(new_chromosome.accuracy * new_chromosome.num_dimensions + 1)
            # Replace all values before the crossover point with the other parent
            for index in range(crossover_point):
                new_chromosome.bit_string[index] = chromosome.bit_string[index]
            return new_chromosome

        def mutate(self, mutate_probability):
            for index in range(self.accuracy * self.num_dimensions):
                if random.random() < mutate_probability:
                    if self.bit_string[index] == 1:
                        self.bit_string[index] = 0
                    else:
                        self.bit_string[index] = 1

    def run(self):
        for fn_params in self.function_list:
            self.population = self.init_population(fn_params)
            grapher = Grapher(fn_params.lower_limit, fn_params.upper_limit)

            for step_num in range(self.max_steps):
                if not self.test_fitness(fn_params):
                    break
                grapher.addValue(self.best_chromosome.chromosome_to_float(), self.best_chromosome.raw_fitness, step_num)
                if step_num == 1:
                    print(fn_params.name)
                    self.print_generation()
                elif step_num == self.max_steps - 1:
                    self.print_generation()

                self.crossover()

                self.mutate()
            grapher.plot(fn_params.fitness_fn, fn_params.name)


    def test_fitness(self, fn_params: FunctionParamsGA):
        lowest_raw_fitness = float("inf")
        highest_raw_fitness = float("-inf")
        highest_location = None
        total_raw_fitness = 0
        for chromosome in self.population:
            # Update the fitness value to be the raw value
            chromosome.eval_fitness(fn_params.fitness_fn)

            # Find the lowest and highest raw fitness values along with the average
            if chromosome.raw_fitness < lowest_raw_fitness:
                lowest_raw_fitness = chromosome.raw_fitness
            if chromosome.raw_fitness > highest_raw_fitness:
                highest_raw_fitness = chromosome.raw_fitness
                highest_location = chromosome
            total_raw_fitness += chromosome.raw_fitness

        # Record raw results
        self.average_raw_fitness = total_raw_fitness / self.population_size
        self.lowest_raw_fitness = lowest_raw_fitness
        self.highest_raw_fitness = highest_raw_fitness
        self.best_chromosome = highest_location

        # Scale fitness values to positive
        total_fitness = 0
        for chromosome in self.population:
            chromosome.fitness = chromosome.raw_fitness - lowest_raw_fitness
            #print(chromosome.fitness)
            total_fitness += chromosome.fitness
        highest_fitness = highest_raw_fitness - lowest_raw_fitness

        # Normalize fitness between bounds
        for chromosome in self.population:
            chromosome.fitness = (chromosome.fitness / highest_fitness) \
                                 * (self.normalize_fitness_high - self.normalize_fitness_low) \
                                 + self.normalize_fitness_low

        # If all fitness values are 0, then they must be identical
        if total_fitness <= 0:
            print("ALL CHROMOSOMES ARE IDENTICAL")
            return False
        return True
            #exit()

    def crossover(self):
        # Create list of parents that is twice the size of the (population - elite clones)
        parent_list = self.getParents()

        elites = self.getEliteClones()
        # Clear the current population for the new generation
        self.population.clear()

        # Add the elites first
        self.population.extend(elites)

        # Crossover each pair of 2 parents appending the children to a new population
        it = iter(parent_list)
        for item in it:
            self.population.append(item.crossover(next(it)))


    # Create list of parents that is twice the size of the (population - elite clones) based on their fitness
    def getParents(self):
        indices = range(0, self.population_size)
        weights = list()
        for individual in self.population:
            weights.append(individual.fitness)
        parent_indices = random.choices(indices, weights=weights, k=(self.population_size - self.elite_clones) * 2)
        parent_list = list()
        for index in parent_indices:
            parent_list.append(self.population[index])
        return parent_list


    # Returns a list of elites from the current population based on their fitness values
    def getEliteClones(self):
        # Find the indices of the n elite
        elite = list()
        # Do once for each elite
        for i in range(self.elite_clones):
            best_index = -1
            # Search through the population for the next best elite
            for index, individual in enumerate(self.population):
                #print("    " + str(individual.raw_fitness))
                # If it's not already elite and it's either the first or better than the previous
                if index not in elite and (best_index == -1 or individual.raw_fitness > self.population[best_index].raw_fitness):
                    #print("        New Elite")
                    best_index = index
            # If we found another elite
            if best_index != -1:
                self.elite_indices[i] = best_index
                elite.append(best_index)
            else:
                print("More elite than population size")

        # Get the actual individuals from the elite indices
        elite_list = list()
        for index in elite:
            print("Elite: " + str(self.population[index].raw_fitness))
            #self.print_chromosomes()
            elite_list.append(self.population[index])

        return elite_list

    # Mutate all non-elite individuals
    def mutate(self):
        for index, individual in enumerate(self.population):
            is_elite = False
            for elite_index in self.elite_indices:
                if index != elite_index:
                    is_elite = True
            if not is_elite:
                individual.mutate(self.mutation_prob)

    def init_population(self, fn_params: FunctionParamsGA):
        pop_list = list()
        for index in range(self.population_size):
            pop_list.append(self.Chromosome(self.num_dimensions, fn_params.lower_limit, fn_params.upper_limit))
        return pop_list

    def print_generation(self):
        print("New Population")
        print("   highest raw fitness: " + str(self.highest_raw_fitness))
        print("   average raw fitness: " + str(self.average_raw_fitness))
        print("   lowest raw fitness:  " + str(self.lowest_raw_fitness) + "\n")

    def print_chromosomes(self):
        for individual in self.population:
            print(str(individual.bit_string) + str(individual.chromosome_to_float()) + "   :   " + str(individual.raw_fitness))

    def print_best(self):
        best = float("-inf")
        for individual in self.population:
            if individual.raw_fitness > best:
                best = individual.raw_fitness
        print("    Best: " + str(best))

ga = GA()
ga.run()
