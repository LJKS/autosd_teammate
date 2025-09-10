import artificial_data
import os
import json

EXPERIMENT_RUNNERS = {'artificial': artificial_data.get_artificial_experiment_runner}

def get_config(argparse_args):
    config_folder = argparse_args.config_folder
    config_json = argparse_args.config_json
    config_path = os.path.join(config_folder, f"{config_json}.json")
    # open config json file
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config
    else:
        raise FileNotFoundError(f"Config file {config_path} not found.")


def get_experiment_runner(config):
    #e.g. config/test_artificial.json
    experiment_runner_name = config['experiment_runner']
    if experiment_runner_name in EXPERIMENT_RUNNERS.keys():
        experiment_runner = EXPERIMENT_RUNNERS[experiment_runner_name]
        experiment_runner_kwargs = config.get('experiment_runner_kwargs', {})
        experiment_runner, variables, info = experiment_runner(**experiment_runner_kwargs)
        return experiment_runner, variables, info
