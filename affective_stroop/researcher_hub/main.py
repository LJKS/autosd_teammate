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
    def experimentalist_on_state(variables, num_samples=general_config['experimentalist_num_samples']):
        value = uniform_sampling(variables, num_samples=num_samples)
        print(f'Sampled conditions: {value}')
        return value

    theorist, theorist_preprocessing = config_mapping.get_theorist(config)

    @on_state(output=['experiment_data'])
    def experiment_runner_on_state(conditions):
        results = experiment_runner(conditions)
        return results

    save_func = config_mapping.get_save_func(config)

    final_state = workflow(theorist, experimentalist_on_state, experiment_runner_on_state, initial_state, cycles=general_config['cycles'], burn_in=general_config['burn_in'],
                           theorist_preprocessing=theorist_preprocessing, save_func=save_func)
    print("Final State Models:", final_state.models[-1])
    print(f"debug:{[type(model) for key,model in final_state.models[-1].items()]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Autora workflow with specified parameters.")
    parser.add_argument('-c', '--config_json', type=str, default='test_artificial', help='Maximum depth of the generated equation.')
    parser.add_argument('-cf', '--config_folder', type=str, default='configs', help='Folder containing configuration files. Default is "./configs".')
    args = parser.parse_args()

    main(args)