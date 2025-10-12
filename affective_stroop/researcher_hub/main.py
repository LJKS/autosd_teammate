import argparse
from autora.state import on_state, StandardState
from autora.variable import VariableCollection, IV, DV
from autora.theorist.bms import BMSRegressor
import artificial_data
from workflow import workflow, uniform_sampling
import config_mapping

def main(argparse_args):
    """
    Run the full experiment workflow, including sampling conditions, executing the experiment, 
    updating theorist models, and saving results.

    :param argparse_args: argparse.Namespace - Parsed command-line arguments used to load the configuration.
    """
    
    # Load configuration from file
    config = config_mapping.get_config(argparse_args)
    general_config = config['general']
    
    # Initialize the experiment runner, variables, and additional info from the config
    experiment_runner, variables, info = config_mapping.get_experiment_runner(config)

    # Initialize the experiment state with variables but no conditions, data, or models yet
    initial_state = StandardState(variables=variables, conditions=None, experiment_data=None, models=[])


    # Define the experimentalist function to sample conditions
    @on_state(output=['conditions'])
    def experimentalist_on_state(variables, num_samples=general_config['experimentalist_num_samples']):
        # Sample conditions uniformly from the variable space
        value = uniform_sampling(variables, num_samples=num_samples)
        print(f'Sampled conditions: {value}')
        return value

    # Initialize the theorist model and its preprocessing function
    theorist, theorist_preprocessing = config_mapping.get_theorist(config)

    # Define the experiment runner function to execute the experiment for given conditions
    @on_state(output=['experiment_data'])
    def experiment_runner_on_state(conditions):
        results = experiment_runner(conditions)
        return results

    # Retrieve the function used to save experiment outputs
    save_func = config_mapping.get_save_func(config)

    # Run the workflow with specified cycles and burn-in, including theorist updates and saving
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