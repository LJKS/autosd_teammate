import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

def save_this_cycle(config, step):
    """
    Determine whether the current workflow step should trigger saving results based on configuration.
    :param config: dict - Configuration dictionary containing:
                   - save: dict with 'save_every' specifying the interval of steps to save.
                   - general: dict with 'cycles' specifying the total number of workflow cycles.
    :param step: int - Current step or cycle number in the workflow.
    :return: bool - True if results should be saved on this step, False otherwise.
    """
    save_every = config['save']['save_every']
    max_steps = config['general']['cycles']
    if step % save_every == 0 or step == max_steps:
        return True
    return False

def get_affective_stroop_phase_summary_visualization_and_serialization(config, save_path):
    def affective_stroop_phase_summary_visualization_and_serialization(state, theorist_preprocessing_function, step):
        if save_this_cycle(config, step):
            #first make sure the path exists, if not create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            timestamp = time.strftime("%Y%m%d-%H%M")
            #create subfolder for current step from step and time
            step_save_path = os.path.join(save_path, f'{timestamp}_step_{step}')
            if not os.path.exists(step_save_path):
                os.makedirs(step_save_path)
            #put the current timestamp into the config dict
            config['evaluation'] = {'timestamp': timestamp, 'step': step}
            #save experiment data
            experiment_data = state.experiment_data
            experiment_data_path = os.path.join(step_save_path, 'experiment_data.csv')
            experiment_data.to_csv(experiment_data_path, index=False)
    return affective_stroop_phase_summary_visualization_and_serialization

