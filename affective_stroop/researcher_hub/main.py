import argparse
from autora.state import on_state, StandardState
from autora.variable import VariableCollection, IV, DV
from autora.theorist.bms import BMSRegressor
import artificial_data
from workflow import workflow, uniform_sampling
import config_mapping

def main(argparse_args):
    config = config_mapping.get_config(argparse_args)
    general_config = config['general']
    experiment_runner, variables, info = config_mapping.get_experiment_runner(config)

    initial_state = StandardState(variables=variables, conditions=None, experiment_data=None, models=[])



    @on_state(output=['conditions'])
    def experimentalist_on_state(variables, num_samples=2):
        value = uniform_sampling(variables, num_samples=num_samples)
        print(f'Sampled conditions: {value}')
        return value

    @on_state(output=['experiment_data'])
    def experiment_runner_on_state(conditions):
        results = experiment_runner(conditions)
        return results

    final_state = workflow(theorist_on_state, experimentalist_on_state, experiment_runner_on_state, initial_state, cycles=general_config['cycles'], burn_in=general_config['burn_in'])
    print("Final State Models:", final_state.models[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Autora workflow with specified parameters.")
    parser.add_argument('-c', '--config_json', type=str, default='test_artificial', help='Maximum depth of the generated equation.')
    parser.add_argument('-cf', '--config_folder', type=str, default='configs', help='Folder containing configuration files. Default is "./configs".')
    args = parser.parse_args()

    main(args)