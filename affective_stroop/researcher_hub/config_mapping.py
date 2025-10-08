import artificial_data
import os
import json
import affective_stroop_firebase_runner
import theorist_zoo
import visualization_and_serialization

SAVE_FREQUENCY_DEFAULT = 5

EXPERIMENT_RUNNERS = {'artificial': artificial_data.get_artificial_experiment_runner,
                      'affective_stroop_firebase': affective_stroop_firebase_runner.get_affective_stroop_firebase_runner}
THEORISTS = {'basic_bms': theorist_zoo.get_bms_theorist}
THEORIST_PREPROCESSING_FUNCTIONS = {'basic_affective_stroop_theorist_preprocessing': theorist_zoo.basic_affective_stroop_theorist_preprocessing,
                                    'none': lambda x: x}
SAVE_FUNCS = {'affective_stroop_phase_summary_visualization_and_serialization': visualization_and_serialization.get_affective_stroop_phase_summary_visualization_and_serialization}
def check_config(config):
    """
    Check if the config is valid.
    :param config: A dict read from the config json file.
    :return: True if and only if the config (superficially) valid.
    Makes sure the following exist:
        - experiment_runner
         - general
         - general.cycles
         - general.burn_in
         - general.save_path
    """
    try:
        _ = config['experiment_runner']
        _ = config['general']
        _ = config['general']['cycles']
        _ = config['general']['burn_in']
        _ = config['general']['save_path']
        return True
    except KeyError as e:
        raise KeyError(f"Config is missing required key: {e}")

def get_config(argparse_args):
    config_folder = argparse_args.config_folder
    config_json = argparse_args.config_json
    config_path = os.path.join(config_folder, f"{config_json}.json")
    # open config json file
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            # check if config is valid
            check_config(config)
            return config
    else:
        raise FileNotFoundError(f"Config file {config_path} not found.")


def get_experiment_runner(config):
    #e.g. config/test_artificial.json
    experiment_runner_config = config['experiment_runner']
    experiment_runner_type = experiment_runner_config['type']
    if experiment_runner_type in EXPERIMENT_RUNNERS.keys():
        experiment_runner = EXPERIMENT_RUNNERS[experiment_runner_type]
        experiment_runner_kwargs = experiment_runner_config.get('kwargs', {})
        experiment_runner, variables, info = experiment_runner(**experiment_runner_kwargs)
        return experiment_runner, variables, info
    else:
        raise ValueError(f"Experiment runner type {experiment_runner_type} not recognized/implemented. Please adjust config_mapping.py to include it.")

def get_theorist(config):
    theorist_config = config['theorist']
    theorist_type = theorist_config["model"]
    theorist = THEORISTS[theorist_type]
    theorist_kwargs = theorist_config.get('model_kwargs', {})
    theorist = theorist(**theorist_kwargs)
    theorist_preprocessing_function_type = theorist_config['preprocessing_function']
    preprocessing_function = THEORIST_PREPROCESSING_FUNCTIONS[theorist_preprocessing_function_type]
    return theorist, preprocessing_function

def get_save_func(config):
    save_config = config['save']
    save_func_factory = SAVE_FUNCS[save_config['type']]
    save_path = config['save']['save_path']
    save_func = save_func_factory(config, save_path)
    return save_func