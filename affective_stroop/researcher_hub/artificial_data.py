from equation_sampler import sample_equations
from autora.experiment_runner.synthetic.abstract.equation import equation_experiment
from autora.variable import IV, DV, VariableCollection
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
    """
    Generate a single random symbolic equation based on the given parameters.
    :param equation_max_depth: int - Maximum depth of the generated equation tree.
    :param equation_num_variables: int - Number of distinct variables to include in the equation.
    :param equation_num_constants: int - Number of distinct constants to include in the equation.
    :return: sympy.Expr - A symbolic equation object representing the generated equation.
    """
    equation = sample_equations(num_samples=1, max_depth=equation_max_depth, max_num_variables=equation_num_variables,
                                 max_num_constants=equation_num_constants,
                                 function_space=FUNCTION_SPACE,
                                 operator_space=OPERATOR_SPACE, verbose=False)
    return equation['sympy_equations'][0]

def get_vars_and_constants(equation):
    """
    Extract and separate variable and constant symbols from a symbolic equation.
    :param equation: sympy.Expr - The symbolic equation from which to extract symbols.
    :return: tuple[list[str], list[str]] - A tuple containing two lists: 
             (1) variable names containing 'x', 
             (2) constant names containing 'c'.
    """
    args = list(equation.free_symbols)
    args = sorted(args, key=lambda el: el.name)
    args = [str(arg) for arg in args]
    variables = [arg for arg in args if 'x' in arg]
    constants = [arg for arg in args if 'c' in arg]
    return variables, constants

def generate_constant_values(constants, value_ranges=(-10,10), random_state_seed=42):
    """
    Generate random numeric values for a list of constant symbols within a specified range.
    :param constants: list[str] - Names of constants for which to generate values.
    :param value_ranges: tuple[float, float] - Lower and upper bounds for the random values (default: (-10, 10)).
    :param random_state_seed: int - Seed for the random number generator to ensure reproducibility (default: 42).
    :return: dict[str, float] - A dictionary mapping each constant name to a randomly generated value.
    """
    random_gen = np.random.default_rng(random_state_seed)
    constant_values = {const: random_gen.uniform(value_ranges[0], value_ranges[1]) for const in constants}
    return constant_values

def equation_with_concrete_constants(equation, constant_values):
    """
    Substitute numeric values for constants in a symbolic equation.
    :param equation: sympy.Expr - The symbolic equation containing constant symbols.
    :param constant_values: dict[str, float] - A mapping of constant names to their numeric values.
    :return: sympy.Expr - The equation with the specified constants replaced by their numeric values.
    """
    if len(constant_values) > 0:
        free_symbols = list(equation.free_symbols)
        symbol_values = {f_sym: constant_values[str(f_sym)] for f_sym in free_symbols if str(f_sym) in list(constant_values.keys())}
        equation = equation.subs(symbol_values)
        return equation
    else:
        return equation

def get_artificial_experiment_runner(equation_max_depth=4,equation_num_variables=2,equation_num_constants=1, value_ranges=(-10,10), num_values=100, random_state_seed=42,
                                     generate_max_attempts=100, constant_gen=generate_constant_values):
    """
    Create an artificial experiment setup using a randomly generated symbolic equation.
    :param equation_max_depth: int - Maximum depth of the generated equation tree (default: 4).
    :param equation_num_variables: int - Number of distinct variables in the equation (default: 2).
    :param equation_num_constants: int - Number of distinct constants in the equation (default: 1).
    :param value_ranges: tuple[float, float] - Range of values for variables and constants (default: (-10, 10)).
    :param num_values: int - Number of discrete values to sample for each independent variable (default: 100).
    :param random_state_seed: int - Seed for the random number generator to ensure reproducibility (default: 42).
    :param generate_max_attempts: int - Maximum number of attempts to generate a valid equation (default: 100).
    :param constant_gen: callable - Function used to generate constant values (default: generate_constant_values).
    :return: tuple[
                equation_experiment, 
                VariableCollection, 
                dict[str, Any]
             ] - A tuple containing:
                (1) the generated experiment object,
                (2) the variable collection (independent and dependent variables),
                (3) a dictionary with metadata including the equation, constants, and their numeric values.
    :raises ValueError: If no valid equation can be generated within the specified number of attempts.
    """
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
    #turn variables into autora variables
    variables = VariableCollection([IV(name=var, value_range=value_ranges) for var in variables], [DV(name='y')])
    #equation, constants, constant_values
    info = {'equation': equation, 'constants': constants, 'constant_values': constant_values}
    return experiment, variables, info



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
