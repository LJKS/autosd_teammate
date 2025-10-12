from autora.state import on_state, StandardState
from autora.variable import VariableCollection, IV, DV
from autora.experimentalist import random as autora_random


def uniform_sampling(variables, num_samples=1):
    """
    Generate uniform random samples from the specified variables.
    :param variables: VariableCollection - Collection of independent and covariate variables to sample from.
    :param num_samples: int - Number of samples to generate (default: 1).
    :return: pandas.DataFrame - DataFrame containing the sampled values for each variable.
    """
    samples = autora_random.pool(variables, num_samples)
    return samples

def experiment_cycle(theorist, experimentalist, experiment_runner, current_state, theorist_preprocessing, save_func, step):
    """
    Run a single experimental-theoretical cycle within the adaptive experiment framework.

    This function represents one iteration of interaction between a theorist (model learner),
    an experimentalist (condition sampler), and an experiment runner (data generator or executor).
    It updates the experiment state based on model predictions, new experimental conditions,
    and collected data, and saves intermediate results.

    :param theorist: callable - Function that updates the theoretical model based on current experiment data.
    :param experimentalist: callable - Function that samples new experimental conditions given the current state.
    :param experiment_runner: callable - Function that runs the experiment under the sampled conditions and returns new data.
    :param current_state: StandardState - The current state of the experiment, containing data, models, and variables.
    :param theorist_preprocessing: callable - Function to preprocess data before feeding it to the theorist.
    :param save_func: callable - Function responsible for saving experiment data and metadata at the current step.
    :param step: int - Current iteration number in the experimental cycle.

    :return: StandardState - Updated experiment state after completing one theorist-experimentalist cycle.
    """
    if current_state.experiment_data is None :
        state = current_state
    else:
        state = theorist(current_state, preprocessing_function=theorist_preprocessing)
        print(f"Theorist models at step {step}:", state.models[-1])

    state = experimentalist(state)
    state = experiment_runner(state)
    save_func(state, theorist_preprocessing_function=theorist_preprocessing, step=step)
    return state

def workflow(theorist, experimentalist, experiment_runner, initial_state, cycles, burn_in, theorist_preprocessing, save_func):
    """
    Run the full adaptive experimental workflow between a theorist and an experimentalist.

    This function orchestrates the iterative interaction between a theorist (model learner),
    an experimentalist (condition sampler), and an experiment runner (data generator) across 
    multiple experimental cycles. It begins with a burn-in phase using random uniform sampling
    to initialize the state with baseline data, and then iteratively refines the process
    through theorist-experimentalist interaction.

    :param theorist: callable - Function that updates the theoretical model based on experiment data.
    :param experimentalist: callable - Function that samples new experimental conditions given the current state.
    :param experiment_runner: callable - Function that executes experiments under given conditions and collects data.
    :param initial_state: StandardState - Initial state of the experiment, including variables and empty data structures.
    :param cycles: int - Total number of experimental cycles to run after the burn-in phase.
    :param burn_in: int - Number of initial samples to generate via random uniform sampling before starting adaptive cycles.
    :param theorist_preprocessing: callable - Function for preprocessing data before feeding it to the theorist.
    :param save_func: callable - Function that saves experiment results and metadata at each cycle.

    :return: StandardState - Final state of the experiment after completing all cycles, 
             containing trained models and collected experimental data.
    """
    #burn in with random uniform sampling
    @on_state(output=['conditions'])
    def on_state_uniform_sampling(variables):
        value = uniform_sampling(variables, num_samples=burn_in)
        return value

    state = experiment_cycle(theorist, on_state_uniform_sampling, experiment_runner, initial_state, theorist_preprocessing=theorist_preprocessing, save_func=save_func, step=0)

    for cycle in range(cycles):
        state = experiment_cycle(theorist, experimentalist, experiment_runner, state, theorist_preprocessing=theorist_preprocessing, save_func=save_func, step=cycle+1)
    return state


def main():
    """
    Run a full artificial experiment workflow using the BMSRegressor theorist.

    This function demonstrates an end-to-end example of the adaptive scientific discovery process.
    It creates an artificial experiment, defines the theorist, experimentalist, and experiment runner
    as state-based functions, and executes a workflow of iterative cycles where models are updated 
    and new experimental conditions are sampled.

    The process starts by generating a synthetic experimental environment from a known equation,
    then alternates between model fitting (theorist) and experimental sampling (experimentalist)
    to iteratively approximate the underlying data-generating process.

    :return: None
        Prints the final trained models to the console.

    **Workflow Overview**:
    1. Initializes a synthetic experiment with a mathematical equation and constants.
    2. Defines:
       - A **theorist** (`theorist_on_state`): fits a BMSRegressor to the collected data.
       - An **experimentalist** (`experimentalist_on_state`): samples new experimental conditions uniformly.
       - An **experiment runner** (`experiment_runner_on_state`): generates experimental data from conditions.
    3. Runs a full workflow via the `workflow()` function for 20 burn-in samples followed by 20 adaptive cycles.
    4. Outputs the final learned model.
    """
    
    from autora.state import on_state, StandardState
    from autora.variable import VariableCollection, IV, DV
    from autora.theorist.bms import BMSRegressor
    import artificial_data
    experiment, variables, info = artificial_data.get_artificial_experiment_runner(equation_max_depth=4, equation_num_variables=2,equation_num_constants=1)
    equation = info['equation']
    constants = info['constants']
    constant_values = info['constant_values']
    initial_state = StandardState(variables=variables, conditions=None, experiment_data=None, models=[])
    @on_state(output=['models'])
    def theorist_on_state(experiment_data, variables, epochs=10):
        bms = BMSRegressor(epochs=10)
        independent_var_names = [var.name for var in variables.independent_variables]
        dependent_var_names = [var.name for var in variables.dependent_variables]
        x = experiment_data[independent_var_names]
        y = experiment_data[dependent_var_names]
        bms.fit(x, y)
        return [bms]

    @on_state(output=['conditions'])
    def experimentalist_on_state(variables, num_samples=1):
        value = uniform_sampling(variables, num_samples=num_samples)
        return value

    @on_state(output=['experiment_data'])
    def experiment_runner_on_state(conditions):
        results = experiment.run(conditions)
        return results

    final_state = workflow(theorist_on_state, experimentalist_on_state, experiment_runner_on_state, initial_state, cycles=20, burn_in=20)
    print("Final State Models:", final_state.models[-1])
def test():
    try:
        main()

    except Exception as e:
        print("Test failed with exception:", e)
        return False
    return True


if __name__ == "__main__":
    test()