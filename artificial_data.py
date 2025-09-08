from autora.experiment_runner.synthetic.abstract.equation import equation_experiment
from equation_sampler import sample_equations
from autora.variable import IV
from autora.experiment_runner.synthetic.abstract.equation import equation_experiment
from autora.variable import IV, DV
import pandas as pd
import numpy as np

FUNCTION_SPACE = [ "sin", "cos","tan","exp","log","sqrt","abs"]

OPERATOR_SPACE = ["+","-","*","/","^"]

def sample_equation(equation_max_depth=4,equation_num_variables=2,equation_num_constants=1):
    equation = sample_equations(num_samples=1, max_depth=equation_max_depth, max_num_variables=equation_num_variables,
                                 max_num_constants=equation_num_constants,
                                 function_space=FUNCTION_SPACE,
                                 operator_space=OPERATOR_SPACE, verbose=False)
    return equation['sympy_equations'][0]

def get_vars_and_constants(equation):
    args = list(equation.free_symbols)
    args = sorted(args, key=lambda el: el.name)
    args = [str(arg) for arg in args]
    variables = [arg for arg in args if 'x' in arg]
    constants = [arg for arg in args if 'c' in arg]
    return variables, constants

def get_artificial_experiment_runner(equation_max_depth=4,equation_num_variables=2,equation_num_constants=1, value_ranges=(-10,10), num_values=100, random_state_seed=42):
    equation = sample_equation(equation_max_depth,equation_num_variables,equation_num_constants)
    variables, constants = get_vars_and_constants(equation)
    num_constants = len(constants)
    num_variables = len(variables)
    independent_vars = [IV(name=var, allowed_values= np.linspace(value_ranges[0], value_ranges[1], num_values), value_range=value_ranges) for var in variables]
    dependent_var = DV(name='y')
    experiment = equation_experiment(equation, independent_vars, dependent_var, random_state=random_state_seed)
    return experiment, equation, variables, constants


if __name__ == "__main__":
    experiment, equation, variables, constants = get_artificial_experiment_runner(equation_max_depth=4,equation_num_variables=2,equation_num_constants=1)
    print("Generated Equation:", equation)
    print("Variables:", variables)
    print("Constants:", constants)
    num_samples = 5
    random_data = np.random.uniform(-10, 10, size=(num_samples, len(variables)))
    test_conditions = pd.DataFrame(random_data, columns=variables)
    if len(constants) > 0:
        for const in constants:
            #set all entries on this constant to random value between -10 and 10
            const_val = np.random.uniform(-10, 10)
            test_conditions[const] = const_val
    print("Test conditions:")
    print(test_conditions)
    print("Experiment Results:")
    print(experiment.run(test_conditions))
