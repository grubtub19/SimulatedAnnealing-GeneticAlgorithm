import numpy as np
import math
import sys
import ga_eval
import random
from FunctionParams import FunctionParamsSA
from Grapher import Grapher


class SA:

    def __init__(self):
        self.__default_dim = 2
        self.num_steps = 5000
        self.max_temp = 50
        num_subdivisions = 40
        significant_digits = 2
        self.schedule_fn = self.schedule_linear_normalized
        self.function_list = [FunctionParamsSA('bump', 0, 5, significant_digits, num_subdivisions, ga_eval.bump,
                                                  ga_eval.bump_c),
                            FunctionParamsSA('sphere', -5, 5, significant_digits, num_subdivisions, ga_eval.sphere, ga_eval.sphere_c),
                            FunctionParamsSA('griew', 0, 200, significant_digits, num_subdivisions, ga_eval.griew, ga_eval.griew_c),

                            FunctionParamsSA('shekel', 0, 10, significant_digits, num_subdivisions, ga_eval.shekel, ga_eval.shekel_c),
                            FunctionParamsSA('micha', -100, 100, significant_digits, num_subdivisions, ga_eval.micha, ga_eval.micha_c),
                            FunctionParamsSA('langermann', 0, 10, significant_digits, num_subdivisions, ga_eval.langermann,
                                                ga_eval.langermann_c),
                            FunctionParamsSA('odd_square', -5 * math.pi, 5 * math.pi, significant_digits, num_subdivisions,
                                                ga_eval.odd_square, ga_eval.odd_square_c)
                            ]

    class SAValues:
        def __init__(self, num_dimensions: int, fn_params):
            self.fn_params = fn_params
            self.float_array = self.getInitialValues(num_dimensions)
            #print("initial: " + str(self.float_array))
            #print("initial: " + str(self.float_array))
            self.current_fitness = self.fn_params.fitness_fn(self.float_array)

        def getInitialValues(self, num_dimensions):
            values = np.array([])
            for dimension in range(num_dimensions):
                values = np.append(values, round(random.uniform(self.fn_params.lower_limit, self.fn_params.upper_limit),
                                    self.fn_params.significant_digits))
            return values

        def chooseSuccessor(self, temp: float):
            successor = np.array([])
            valid_successor = False
            failed_moves = list()
            num_fails = 0

            # Successor means positive fitness or probability
            while not valid_successor:
                valid_neighbor = False
                #print("valid start")
                # Look for new neighbor within the bounds
                while not valid_neighbor:
                    successor = self.float_array.copy()
                    num = random.randrange(self.float_array.shape[0])
                    dir = random.randrange(-1, 2, 2)
                    #print(dir)
                    successor[num] = self.float_array[num] + (self.fn_params.step_size * dir)
                    valid_neighbor = self.fn_params.costraint_fn(successor)
                    # if the neighbor is out of bounds, add it to the failures list if its not already there
                    if not valid_neighbor:
                        equal_found = False
                        for failure in failed_moves:
                            if np.array_equal(successor, failure[0]):
                                equal_found = True
                        if not equal_found:
                            failed_moves.append([successor, self.fn_params.fitness_fn(successor)])
                            #print('append')

                repeat_copy = False
                #print("len(failed moves): " + str(len(failed_moves)))
                for failure in failed_moves:

                    if np.array_equal(successor, failure[0]):
                        #print("Already Failed")
                        # print(failure[0])
                        # print(successor)
                        repeat_copy = True
                #print(str(self.float_array))
                #print((str(successor)))
                #print(str(failed_moves))
                if len(failed_moves) == 2 * self.float_array.shape[0]:
                    num_fails += 1
                    #print("Out of options")
                    #successor = self.float_array
                    valid_successor = True

                if not repeat_copy:
                    successor_fitness = self.fn_params.fitness_fn(successor)
                    fitness_delta = successor_fitness - self.current_fitness
                    #fitness_delta =  self.current_fitness - successor_fitness
                    # If it's a good move or we accept the bad move randomly based on the temp
                    #print("fitness_delta: " + str(fitness_delta))
                    #print("temp: " + str(temp))
                    #print("probability: " + str(math.exp(fitness_delta / temp)))

                    if fitness_delta > 0:
                        valid_successor = True
                        #print(" good_move")
                    elif random.random() <= math.exp(fitness_delta / temp):
                        #print(" bad_move")
                        valid_successor = True
                    else:
                        failed_moves.append([successor, successor_fitness])
            #print("CHOSEN")
            #print(num_fails)
            self.float_array = successor
            self.current_fitness = self.fn_params.fitness_fn(self.float_array)

    def run(self):
        for fn_params in self.function_list:
            step_num = 0
            sa_values = self.SAValues(self.__default_dim, fn_params)
            temp = self.schedule_fn(step_num)
            grapher = Grapher(fn_params.lower_limit, fn_params.upper_limit)
            #print(fn_params.name)
            #print("Initial Fitness: " + str(sa_values.current_fitness))
            while not temp == 0:
                sa_values.chooseSuccessor(temp)
                grapher.addValue(sa_values.float_array, sa_values.current_fitness, step_num)

                step_num += 1
                temp = self.schedule_fn(step_num)
            #print("Final Fitness:   " + str(sa_values.current_fitness))
            grapher.plot(fn_params.fitness_fn, fn_params.name)

    def schedule_linear(self, step_num):
        return self.num_steps - step_num - 1

    def schedule_linear_normalized(self, step_num):
        return (self.num_steps - step_num - 1) / (self.num_steps / self.max_temp)

sa = SA()
sa.run()