"""
Basic Workflow
    Single condition Variable (0-1), Single Observation Variable(0-1)
    Theorist: LinearRegression
    Experimentalist: Random Sampling
    Runner: Firebase Runner (no prolific recruitment)
"""

import json

from autora.variable import VariableCollection, Variable, ValueType
from autora.experimentalist.random import pool
from autora.experiment_runner.firebase_prolific import firebase_runner
from autora.state import StandardState, on_state, Delta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Add the project root to sys.path
import sys
import os

from triton.language import dtype

#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.append(project_root)
# Step 1: Construct absolute path to `testing_zone/src`
stimuli_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../testing_zone/src"))
sys.path.append(stimuli_root)

# Step 2: Now import using `design.stimuli.trial_sequence`
from design.stimuli.trial_sequence import trial_sequence
#from testing_zone.design.stimuli.trial_sequence import trial_sequence
#from testing_zone.src.design.stimuli.trial_sequence import neutral_words, emotional_words


# *** Set up variables *** #       # length and phase

# *** State *** #
# With the variables, we can set up a state. The state object represents the state of our
# closed loop experiment.

# *** Components/Agents *** #
# Components are functions that run on the state. The main components are:
# - theorist
# - experiment-runner
# - experimentalist
# See more about components here: https://autoresearch.github.io/autora/


# ** Theorist ** #
# Here we use a linear regression as theorist, but you can use other theorists included in
# autora (for a list: https://autoresearch.github.io/autora/theorist/)

#theorist = LinearRegression()

# To use the theorist on the state object, we wrap it with the on_state functionality and return a
# Delta object.
# Note: The if the input arguments of the theorist_on_state function are state-fields like
# experiment_data, variables, ... , then using this function on a state object will automatically
# use those state fields.
# The output of these functions is always a Delta object. The keyword argument in this case, tells
# the state object witch field to update.

"""
@on_state()
def theorist_on_state(experiment_data, variables):
    ivs = [iv.name for iv in variables.independent_variables]
    dvs = [dv.name for dv in variables.dependent_variables]
    x = experiment_data[ivs]
    y = experiment_data[dvs]
    return Delta(models=[theorist.fit(x, y)])

# ** Experimentalist ** #
# Here, we use a random pool and use the wrapper to create a on state function
# Note: The argument num_samples is not a state field. Instead, we will pass it in when calling
# the function


@on_state()
def experimentalist_on_state(variables, num_samples):
    return Delta(conditions=pool(variables, num_samples))
"""

# ** Experiment Runner ** #
# We will run our experiment on firebase and need credentials. You will find them here:
# (https://console.firebase.google.com/)
#   -> project -> project settings -> service accounts -> generate new private key



# accuracy and mean RT are computed per phase
def compute_phase_metrics(df):
    #import time#get time string
    #time_str = time.strftime("%Y%m%d-%H%M%S")
    #df.to_csv(f'debug_full_data_{time_str}.csv', index=False)
    #df is a dataframe with important columns:
    # - rt, phase, accuracy (i.e. if correct), valence
    #analysis steps:
    # - group by phase
    # - throw out rows without a given phase (phase is NaN)
    # throw out rows with phase equals  practice
    # - group by phase
    # per phase:
    # - accuracy: mean of correct
    # - assert the valence for all entries in this phase is the same, if not raise error
    # compute mean RT for correct trials, fallse trials, all trials
    #return dataframe with single row, with three rows (1 per phase) with entries:
    # phase, accuracy, mean_rt, n_trials, valence

    #pick the relevant columns
    # Select relevant columns
    dfs_by_participant = [df_participant for _, df_participant in df.groupby("participant_id")]
    def participant_to_phase_summary(participant_df):
        participant_df = participant_df[["phase", "correct", "rt", "valence"]]
        participant_df = participant_df.dropna(subset=["phase"])
        participant_df = participant_df[participant_df["phase"] != "practice"]
        participant_df = participant_df.dropna(subset=["correct", "rt"])

        participant_df["correct"] = participant_df["correct"].astype(bool)
        participant_df["rt"] = pd.to_numeric(participant_df["rt"], errors="coerce")

        def phase_agg(group):
            accuracy = group["correct"].mean()
            mean_rt = group["rt"].mean()
            mean_rt_correct = group.loc[group["correct"], "rt"].mean()
            mean_rt_incorrect = group.loc[~group["correct"], "rt"].mean()
            n_trials = group["rt"].count()
            valence = group["valence"].iloc[0] if group["valence"].nunique() == 1 else "mixed"
            return pd.Series({
                "accuracy": accuracy,
                "mean_rt": mean_rt,
                "mean_rt_correct": mean_rt_correct,
                "mean_rt_incorrect": mean_rt_incorrect,
                "n_trials": n_trials,
                "valence": valence
            })

        participant_phase_summary = participant_df.groupby("phase").apply(phase_agg).reset_index()
        print(participant_phase_summary)
        if (participant_phase_summary["valence"] == "mixed").any():
            raise ValueError("Valence is not consistent within at least one phase.")
        print(participant_phase_summary)
        #rename the n_trials column to length due to legacy code
        participant_phase_summary = participant_phase_summary.rename(columns={"n_trials": "length"})
        print('phase summary:', participant_phase_summary)
        return participant_phase_summary
    participant_dfs = [participant_to_phase_summary(p_df) for p_df in dfs_by_participant]
    phase_summary = pd.concat(participant_dfs, ignore_index=True)
    return phase_summary
    """
    df = df.copy()

    if exclude_practice:
        df = df[df["phase"] != "practice"]

    df["correct"] = df["correct"].astype(bool)
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

    # metrics per phase
    phase_summary = (
        df.groupby("phase")
        .agg(
            accuracy=("correct", "mean"),
            mean_rt=("rt", "mean"),
            n_trials=("rt", "count")
        )
        .reset_index()
    )

    return phase_summary
    """

# parse the data
def parse_raw_experiment_data(raw_data):
    all_trial_data = []

    for item in raw_data:
        try:
            data_obj = json.loads(item)
            participant_id = data_obj.get("id", "unknown")
            n_trials = data_obj.get("n_trials", None)

            # trial-by-trial data
            trials = data_obj.get("trials", [])

            # Convert trials to DataFrame temporarily to compute metrics
            df_trials = pd.DataFrame(trials)
            if not df_trials.empty:
                df_trials["correct"] = df_trials["correct"].astype(bool)
                df_trials["rt"] = pd.to_numeric(df_trials["rt"], errors="coerce")
                accuracy = df_trials["correct"].mean()
                mean_rt = df_trials["rt"].mean()
            else:
                accuracy = None
                mean_rt = None

            # Enrich each trial with metadata
            for trial in trials:
                trial["participant_id"] = participant_id
                trial["n_trials"] = n_trials
                trial["accuracy"] = accuracy
                trial["mean_rt"] = mean_rt

            all_trial_data.extend(trials)

        except Exception as e:
            print(f"❌ Error parsing item: {e}")
            continue

    df = pd.DataFrame(all_trial_data)
    return df

def get_affective_stroop_firebase_runner(firebase_credentials_path: str="firebase_credentials.json",
                                         preprocessing: str="raw",
                                         length_range: tuple=(10, 80)):
    """
    Get an experiment runner that runs the affective stroop task on firebase.
    :param firebase_credentials_path: local path to the firebase credentials json file
    :param preprocessing: either raw or phase_summary
    :return:
    """
    firebase_credentials = json.load(open(firebase_credentials_path))

    # simple experiment runner that runs the experiment on firebase
    firebase_experiment = firebase_runner(
        firebase_credentials=firebase_credentials,
        time_out=100,
        sleep_time=5)
    info= "Lieber Leon, wir hoffen du wirst hiermit glücklich:D LG"

    if preprocessing == "phase_summary":
        variables = VariableCollection(
            independent_variables=[
                Variable(name="length", value_range=length_range, allowed_values=list(range(length_range[0], length_range[1]+1)), type=ValueType.INTEGER)
            ],
            covariates=[
                Variable(name="phase", allowed_values=["neutral1", "emotional", "neutral2"]),
                Variable(name="valence", allowed_values=["neutral", "emotional"]),
                Variable(name="accuracy", value_range=(0, 1))
            ],
            dependent_variables=[
                Variable(name="mean_rt_correct", value_range=(0, 5000)),
                Variable(name="mean_rt_incorrect", value_range=(0, 5000)),
                Variable(name="mean_rt", value_range=(0, 5000))
            ]
        )
    elif preprocessing == "raw":
        #TODO AHHHHHHHHHHHHHHHHHHHHHHHH
        raise NotImplementedError("Raw data variable collection not implemented yet.")
    else:
        raise ValueError(f"Preprocessing {preprocessing} not recognized. Use either 'raw' or 'phase_summary'.")


    def experiment_runner(conditions):
        js_codes = []
        for _, row in conditions.iterrows():
            length = int(row["length"])
            js_code = trial_sequence(length)
            js_codes.append(js_code)
        # these are the conditions we send to firebase (only the experiment code), more information than just length
        conditions_to_send = conditions.copy()
        conditions_to_send["experiment_code"] = js_codes
        # raw data: u.a. trials(enthält alle recordings vom participant wie acc, rt)
        raw_data = firebase_experiment(conditions_to_send)
        # parse and compute phase from raw data
        df = parse_raw_experiment_data(raw_data)
        phase_summary = compute_phase_metrics(df)
        '''
        example phase summary: 
        phase	    accuracy	mean_rt	n_trials
        neutral1	0.85	    634.2	10
        emotional	0.78	    712.5	10
        neutral2	0.88	    645.1	10
        '''
        print('?????????????')
        if preprocessing == "raw":
            print('results!!!!!!!!!!!!', df)
            #save df to csv
            df.to_csv('results.csv', index=False)
            return df
        elif preprocessing == "phase_summary":
            return phase_summary
        else:
            raise ValueError(f"Preprocessing {preprocessing} not recognized. Use either 'raw' or 'phase_summary'.")

    return experiment_runner, variables, info


"""
# Now, we can run our components
for _ in range(3):
    state = experimentalist_on_state(state, num_samples=2)  # Collect 2 conditions per iteration
    state = runner_on_state(state)
    state = theorist_on_state(state)


# *** Report the data *** #
# If you changed the theorist, also change this part
def report_linear_fit(m: LinearRegression, precision=4):
    s = f"y = {np.round(m.coef_[0].item(), precision)} x " \
        f"+ {np.round(m.intercept_.item(), 4)}"
    return s


print(report_linear_fit(state.models[0]))
print(report_linear_fit(state.models[-1]))
"""
