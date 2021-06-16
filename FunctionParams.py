

class FunctionParamsSA:
    def __init__(self, name: str, lower_limit: float, upper_limit: float, significant_digits: int,
                 num_subdivisions: int, fitness_fn: callable, constraint_fn: callable, num_steps: int = 1000):
        self.name = name
        self.fitness_fn = fitness_fn
        self.costraint_fn = constraint_fn
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.significant_digits = significant_digits
        self.num_steps = num_steps
        self.step_size = round(((upper_limit - lower_limit) / num_subdivisions), self.significant_digits)

class FunctionParamsGA:
    def __init__(self, name: str, lower_limit: float, upper_limit: float, fitness_fn: callable, constraint_fn: callable, num_steps: int = 1000):
        self.name = name
        self.fitness_fn = fitness_fn
        self.costraint_fn = constraint_fn
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.num_steps = num_steps