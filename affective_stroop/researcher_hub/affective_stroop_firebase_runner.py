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

# accuracy and mean RT are computed per phase
def compute_phase_metrics(df):
    """
    Compute behavioral performance metrics per phase for each participant in an experimental dataset.
    :param df: pandas.DataFrame - Input dataframe containing at least the following columns:
                'participant_id', 'phase', 'correct', 'rt', and 'valence'.
    :return: pandas.DataFrame - A dataframe summarizing performance per participant and phase, 
             with the following columns:
                - phase: str - Name of the experimental phase.
                - accuracy: float - Mean accuracy (proportion of correct responses).
                - mean_rt: float - Mean reaction time across all trials.
                - mean_rt_correct: float - Mean reaction time for correct trials.
                - mean_rt_incorrect: float - Mean reaction time for incorrect trials.
                - length: int - Number of valid trials in the phase (renamed from n_trials).
                - valence: str - Valence label for the phase (must be consistent within each phase).
    :raises ValueError: If valence values are not consistent within a phase for any participant.
    """
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
        """
        Compute phase-level behavioral metrics for a single participant.
        :param participant_df: pandas.DataFrame - Dataframe containing the participant’s trial-level data with 
            columns 'phase', 'correct', 'rt', and 'valence'.
        :return: pandas.DataFrame - A dataframe summarizing the participant’s performance per phase, 
                with the following columns:
                    - phase: str - Name of the experimental phase.
                    - accuracy: float - Mean accuracy (proportion of correct trials).
                    - mean_rt: float - Mean reaction time across all valid trials.
                    - mean_rt_correct: float - Mean reaction time for correct trials only.
                    - mean_rt_incorrect: float - Mean reaction time for incorrect trials only.
                    - length: int - Number of valid trials in the phase (renamed from n_trials for compatibility).
                    - valence: str - Valence label for the phase; must be consistent, otherwise raises an error.
        :raises ValueError: If multiple valence values are found within the same phase (i.e., valence is not consistent).
        """
        participant_df = participant_df[["phase", "correct", "rt", "valence"]]
        participant_df = participant_df.dropna(subset=["phase"])
        participant_df = participant_df[participant_df["phase"] != "practice"]
        participant_df = participant_df.dropna(subset=["correct", "rt"])

        participant_df["correct"] = participant_df["correct"].astype(bool)
        participant_df["rt"] = pd.to_numeric(participant_df["rt"], errors="coerce")

        def phase_agg(group):
            """
            Aggregate trial-level data within a single experimental phase to compute summary metrics.
            :param group: pandas.DataFrame - Subset of a participant’s data corresponding to one experimental phase,
                containing the columns 'correct', 'rt', and 'valence'.
            :return: pandas.Series - A series containing phase-level summary statistics:
                        - accuracy: float - Mean proportion of correct trials.
                        - mean_rt: float - Mean reaction time across all trials.
                        - mean_rt_correct: float - Mean reaction time for correct trials only.
                        - mean_rt_incorrect: float - Mean reaction time for incorrect trials only.
                        - n_trials: int - Number of valid trials in the phase.
                        - valence: str - Valence for the phase ('mixed' if multiple valence values are present).
            """
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
        if (participant_phase_summary["valence"] == "mixed").any():
            raise ValueError("Valence is not consistent within at least one phase.")
        #rename the n_trials column to length due to legacy code
        participant_phase_summary = participant_phase_summary.rename(columns={"n_trials": "length"})
        return participant_phase_summary
    participant_dfs = [participant_to_phase_summary(p_df) for p_df in dfs_by_participant]
    phase_summary = pd.concat(participant_dfs, ignore_index=True)
    return phase_summary

# parse the data
def parse_raw_experiment_data(raw_data):
    """
    Parse raw JSON-formatted experiment data into a structured pandas DataFrame with enriched trial-level information.
    :param raw_data: list[str] - List of JSON strings, each representing a participant’s experiment data.
    :return: pandas.DataFrame - DataFrame containing one row per trial, including:
                - participant_id: str - Identifier for the participant (default 'unknown' if missing).
                - n_trials: int - Total number of trials for the participant.
                - correct: bool - Whether the trial response was correct.
                - rt: float - Reaction time for the trial.
                - accuracy: float - Participant-level mean accuracy across trials.
                - mean_rt: float - Participant-level mean reaction time across trials.
                - plus any additional fields present in the original trial data.
    """
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
        raise NotImplementedError("Raw data variable collection not implemented yet.")
    else:
        raise ValueError(f"Preprocessing {preprocessing} not recognized. Use either 'raw' or 'phase_summary'.")


    def experiment_runner(conditions):
        """
        Run an experiment for the given conditions, collect trial-level data, and optionally compute phase-level summaries.
        :param conditions: pandas.DataFrame - DataFrame specifying the experimental conditions for each participant/sequence,
                        must include a 'length' column indicating the number of trials per condition.
        :return: tuple[callable, VariableCollection, dict] - A tuple containing:
                    (1) experiment_runner: a function that executes the experiment and returns either raw or phase-level data,
                    (2) variables: VariableCollection object describing independent and dependent variables,
                    (3) info: dict containing metadata such as the generated equation, constants, and constant values.
        
        The experiment_runner function, when called, performs the following steps:
        1. Generates trial sequences for each condition.
        2. Sends conditions to the experiment platform (e.g., Firebase) and collects raw trial data.
        3. Parses raw data into a structured pandas DataFrame using parse_raw_experiment_data.
        4. Computes phase-level metrics using compute_phase_metrics.
        5. Returns either the raw trial DataFrame or the phase-level summary depending on the `preprocessing` argument:
            - 'raw': returns the full trial-level DataFrame and saves it as 'results.csv'.
            - 'phase_summary': returns a summary DataFrame with metrics per phase (accuracy, mean_rt, n_trials).
        :raises ValueError: If the `preprocessing` argument is not recognized.
        """
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
        if preprocessing == "raw":
            #save df to csv
            df.to_csv('results.csv', index=False)
            return df
        elif preprocessing == "phase_summary":
            return phase_summary
        else:
            raise ValueError(f"Preprocessing {preprocessing} not recognized. Use either 'raw' or 'phase_summary'.")

    return experiment_runner, variables, info

