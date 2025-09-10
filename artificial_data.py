from equation_sampler import sample_equations
from autora.experiment_runner.synthetic.abstract.equation import equation_experiment
from autora.variable import IV, DV
import pandas as pd
import numpy as np
import sympy as sp

FUNCTION_SPACE = [ "sin", "cos","tan","exp","log","sqrt","abs"]

OPERATOR_SPACE = ["+","-","*","/","^"]

def check_values_in_domain(equation, value_range):
    """
    Check if the equation is defined for all values in the given range.
    :param equation: sympy equation
    :param value_range: tuple of (min, max)
    :return: True if the equation is defined for all values in the range, False otherwise.
    """
    target_domain = sp.Interval(value_range[0], value_range[1])
    symbols = list(equation.free_symbols)
    print(f'Equation: {equation}, Symbols: {symbols}')
    for symbol in symbols:
        #obtain the domain of the symbol in the reals
        symbol_domain = sp.calculus.util.continuous_domain(equation, symbol, sp.S.Reals)
        print(f'Symbol: {symbol}, Domain: {symbol_domain}')
        #check if value range is in the domain
        if not target_domain.is_subset(symbol_domain):
            return False
    return True

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

def generate_constant_values(constants, value_ranges=(-10,10), random_state_seed=42):
    random_gen = np.random.default_rng(random_state_seed)
    constant_values = {const: random_gen.uniform(value_ranges[0], value_ranges[1]) for const in constants}
    return constant_values

def equation_with_concrete_constants(equation, constant_values):
    if len(constant_values) > 0:
        free_symbols = list(equation.free_symbols)
        symbol_values = {f_sym: constant_values[str(f_sym)] for f_sym in free_symbols if str(f_sym) in list(constant_values.keys())}
        equation = equation.subs(symbol_values)
        return equation
    else:
        return equation

def get_artificial_experiment_runner(equation_max_depth=4,equation_num_variables=2,equation_num_constants=1, value_ranges=(-10,10), num_values=100, random_state_seed=42,
                                     generate_max_attempts=100, constant_gen=generate_constant_values):
    for attempt in range(generate_max_attempts):
        equation = sample_equation(equation_max_depth,equation_num_variables,equation_num_constants)
        print('Trying equation:', equation)
        if check_values_in_domain(equation, value_ranges):
            break
        else:
            if attempt == generate_max_attempts - 1:
                raise ValueError(f'Could not generate a valid equation in {generate_max_attempts} attempts.')
            continue
    print(f'Generated equation in {attempt+1} attempts: {equation}')
    variables, constants = get_vars_and_constants(equation)
    independent_vars = [IV(name=var, allowed_values= np.linspace(value_ranges[0], value_ranges[1], num_values), value_range=value_ranges) for var in variables]
    dependent_var = DV(name='y')
    constant_values = constant_gen(constants, value_ranges, random_state_seed)
    print(equation, constant_values)
    equation = equation_with_concrete_constants(equation, constant_values)
    print(f'equation after substituting constants: {equation}')
    experiment = equation_experiment(equation, independent_vars, dependent_var, random_state=random_state_seed)
    return experiment, equation, variables, constants, constant_values



def test():

    """
    Test the artificial experiment runner by generating a random equation and running it on random test conditions.
    :return: True if the test passes, False otherwise.
    """
    try:
        experiment, equation, variables, constants = get_artificial_experiment_runner(equation_max_depth=4, equation_num_variables=2, equation_num_constants=1)
        num_samples = 5
        random_data = np.random.uniform(-10, 10, size=(num_samples, len(variables)))
        test_conditions = pd.DataFrame(random_data, columns=variables)
        if len(constants) > 0:
            for const in constants:
                # set all entries on this constant to random value between -10 and 10
                const_val = np.random.uniform(-10, 10)
                test_conditions[const] = const_val
        #run the experiment
        results = experiment.run(test_conditions)
        print("Generated Equation:", equation)
        print("Variables:", variables)
        print("Constants:", constants)
        print("Test conditions:")
        print(test_conditions)
        print("Experiment Results:")
        print(results)
    except Exception as e:
        print("Test failed with exception:", e)
        return False
    return True


if __name__ == "__main__":
    experiment, equation, variables, constants = get_artificial_experiment_runner(equation_max_depth=4,equation_num_variables=2,equation_num_constants=1)
    print("Generated Equation:", equation)
    print("Variables:", variables)
    print("Constants:", constants)
    num_samples = 5
    random_data = np.random.uniform(-10, 10, size=(num_samples, len(variables)))
    test_conditions = pd.DataFrame(random_data, columns=variables)
    print("Test conditions:")
    print(test_conditions)
    print("Experiment Results:")
    print(experiment.run(test_conditions))
