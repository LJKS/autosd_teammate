import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

def save_this_cycle(config, step):
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

if __name__ == "__main__":
    #load experiment_data.csv as df
    df = pd.read_csv('experiment_data.csv')
    #looks like:

    """
    phase,accuracy,mean_rt,mean_rt_correct,mean_rt_incorrect,length,valence
    emotional, 0.8, 1219.2, 1299.0, 900.0, 5, emotional
    neutral1, 0.8, 832.8, 890.25, 603.0, 5, neutral
    neutral2, 1.0, 1143.8, 1143.8,, 5, neutral
    emotional, 0.8333333333333334, 928.6666666666666, 885.8, 1143.0, 6, emotional
    neutral1, 1.0, 2940.1666666666665, 2940.1666666666665,, 6, neutral
    neutral2, 0.8333333333333334, 1676.6666666666667, 1787.0, 1125.0, 6, neutral
    emotional, 1.0, 645.2857142857143, 645.2857142857143,, 7, emotional
    neutral1, 1.0, 594.4285714285714, 594.4285714285714,, 7, neutral
    neutral2, 0.8571428571428571, 555.7142857142857, 558.3333333333334, 540.0, 7, neutral
    """
    #jointplot: x = length, y = mean_rt, hue = valence, style = phase
    #marginal distribution only for reation time, separated by valence
    #marginal distribution as kde
    #save to drive
    plots = []
    for rt in ['mean_rt', 'mean_rt_correct', 'mean_rt_incorrect']:
        save_path = 'result.png'
        sns.set(style="whitegrid")
        g = sns.JointGrid(data=df, x="length", y=rt, hue="phase", height=8)
        g.plot_joint(sns.scatterplot, data=df)
        sns.kdeplot(data=df, y=rt, hue="phase", ax=g.ax_marg_y)
        g.ax_joint.set_title(f'{rt}vs. number of trials ')
        g.ax_marg_x.set_visible(False)  # Hide top marginal plot
        g.ax_joint.set_xlabel('Number of trials in phase')
        g.ax_joint.set_ylabel(f'{rt} (ms)')
        g.ax_joint.set_ylim(bottom=0)  # Force y-axis to start at 0
        plt.savefig(f'{rt}_vs_length.png', bbox_inches='tight')
        plots.append(f'{rt}_vs_length.png')
        plt.close()

    plots = []
    for rt in ['mean_rt', 'mean_rt_correct', 'mean_rt_incorrect']:
        save_path = 'result.png'
        sns.set(style="whitegrid")
        g = sns.JointGrid(data=df, x="length", y=rt, hue="phase", height=8)
        g.plot_joint(sns.scatterplot, data=df)
        sns.kdeplot(data=df, y=rt, hue="phase", ax=g.ax_marg_y)
        g.ax_joint.set_title(f'{rt}vs. number of trials ')
        g.ax_marg_x.set_visible(False)  # Hide top marginal plot
        #draw the mean of each phase as a line, colored by phase
        means = df.groupby('phase')[rt].mean()
        for phase, mean in means.items():
            g.ax_joint.axhline(mean, color=sns.color_palette()[list(means.index).index(phase)], linestyle='--', label=f'{phase} mean')
        g.ax_joint.set_xlabel('Number of trials in phase')
        g.ax_joint.set_ylabel(f'{rt} (ms)')
        g.ax_joint.set_ylim(bottom=0)  # Force y-axis to start at 0
        plt.savefig(f'{rt}_vs_length_with_model.png', bbox_inches='tight')
        plots.append(f'{rt}_vs_length.png')
        plt.close()
    #clear all plots
    #accuracy vs length now
    sns.set(style="whitegrid")
    #use opaqueness for overlapping points
    g = sns.JointGrid(data=df, x="length", y='accuracy', hue="phase", height=8)
    g.plot_joint(sns.scatterplot, data=df, alpha=0.7)
    sns.kdeplot(data=df, y='accuracy', hue="phase", ax=g.ax_marg_y)
    g.ax_joint.set_title(f'accuracy vs. number of trials ')
    g.ax_marg_x.set_visible(False)  # Hide top marginal plot
    g.ax_joint.set_xlabel('Number of trials in phase')
    g.ax_joint.set_ylabel(f'Accuracy')
    g.ax_joint.set_ylim(bottom=0)  # Force y-axis to start at 0
    plt.savefig(f'acc_vs_length.png', bbox_inches='tight')
    plt.close()