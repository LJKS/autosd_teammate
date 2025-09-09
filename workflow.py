from autora.state import on_state, StandardState
from autora.variable import VariableCollection, IV, DV
from autora.experimentalist import random as autora_random


def uniform_sampling(variables, num_samples=1):
    samples = autora_random.pool(variables, num_samples)
    return samples

def experiment_cycle(theorist, experimentalist, experiment_runner, current_state, sample_datapoints=1):
    """
    Run a single cycle between a theorist and an experimentalist.
    """
    if current_state.experiment_data is None :
        state = current_state
    else:
        state = theorist(current_state)

    state = experimentalist(state)
    state = experiment_runner(state)
    return state

def workflow(theorist, experimentalist, experiment_runner, initial_state, cycles, burn_in):
    """
    Run a workflow between a theorist and an experimentalist.
    """
    #burn in with random uniform sampling
    @on_state(output=['conditions'])
    def on_state_uniform_sampling(variables):
        value = uniform_sampling(variables, num_samples=burn_in)
        return value

    state = experiment_cycle(theorist, on_state_uniform_sampling, experiment_runner, initial_state, sample_datapoints=burn_in)

    for cycle in range(cycles):
        state = experiment_cycle(theorist, experimentalist, experiment_runner, state)
    return state


def main():
    from autora.state import on_state, StandardState
    from autora.variable import VariableCollection, IV, DV
    from autora.theorist.bms import BMSRegressor
    import artificial_data
    experiment, equation, variables, constants, constant_values = artificial_data.get_artificial_experiment_runner(equation_max_depth=4, equation_num_variables=2,equation_num_constants=1)
    variables = VariableCollection([IV(name=var, value_range=(-10.,10.)) for var in variables], [DV(name='y')])
    initial_state = StandardState(variables=variables, conditions=None, experiment_data=None, models=[])
    print(f'Equation: {equation} with constant values {constant_values}')
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

    final_state = workflow(theorist_on_state, experimentalist_on_state, experiment_runner_on_state, initial_state, cycles=5, burn_in=5)
    print("Final State Models:", final_state.models[-1])
    print(equation)

def test():
    main()
    try:
        main()

    except Exception as e:
        print("Test failed with exception:", e)
        return False
    return True


if __name__ == "__main__":
    test()