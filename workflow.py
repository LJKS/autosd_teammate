def uniform_sampling(state, num_samples):
    state = #TODO:
    """Sample uniformly from the state space."""
    return state

def experiment_cycle(theorist, experimentalist, experiment_runner, current_state, sample_datapoints=1):
    """Run a single cycle between a theorist and an experimentalist.

    """
    #TODO runs one step with the specified theorist, experimentalist, and experiment runner
    return state

def workflow(theorist, experimentalist, experiment_runner, initial_state, cycles, burn_in):
    """Run a workflow between a theorist and an experimentalist.

    """

    #TODO: Prepate the workflow cycle
    #burn in with random uniform sampling
    state = experiment_cycle(theorist, experimentalist, experiment_runner, initial_state, sample_datapoints=burn_in)

    for cycle in range(cycles):
        state = experiment_cycle(theorist, experimentalist, experiment_runner, state)
    return state
