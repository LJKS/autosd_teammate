from autora.state import on_state, StandardState
from autora.variable import VariableCollection, IV, DV
from autora.experimentalist import random as autora_random
"""
Basic Workflow
    Single condition Variable (0-1), Single Observation Variable(0-1)
    Theorist: LinearRegression
    Experimentalist: Random Sampling
    Runner: Firebase Runner (no prolific recruitment)
"""

import json

from autora.variable import VariableCollection, Variable
from autora.experimentalist.random import pool
from autora.experiment_runner.firebase_prolific import firebase_runner
from autora.state import StandardState, on_state, Delta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Add the project root to sys.path
import sys
import os
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.append(project_root)

# Step 1: Construct absolute path to `testing_zone/src`
stimuli_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../testing_zone/src"))
sys.path.append(stimuli_root)

# Step 2: Now import using `design.stimuli.trial_sequence`
from design.stimuli.trial_sequence import trial_sequence
#from testing_zone.design.stimuli.trial_sequence import trial_sequence
#from testing_zone.src.design.stimuli.trial_sequence import neutral_words, emotional_words

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
    print(equation)

def test():
    try:
        main()

    except Exception as e:
        print("Test failed with exception:", e)
        return False
    return True


if __name__ == "__main__":
    test()