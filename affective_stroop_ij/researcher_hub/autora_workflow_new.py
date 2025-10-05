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


# *** Set up variables *** #       # length and phase
variables = VariableCollection(
    independent_variables=    [
        Variable(name="length", value_range=(10, 80)),
        Variable(name="phase", allowed_values=["neutral1", "emotional", "neutral2"])
    ],
    dependent_variables=[Variable(name="accuracy", value_range=(0, 1)),
                        Variable(name="mean_rt", value_range=(0, 5000))])

# *** State *** #
# With the variables, we can set up a state. The state object represents the state of our
# closed loop experiment.
state = StandardState(
    variables=variables,
)

# *** Components/Agents *** #
# Components are functions that run on the state. The main components are:
# - theorist
# - experiment-runner
# - experimentalist
# See more about components here: https://autoresearch.github.io/autora/


# ** Theorist ** #
# Here we use a linear regression as theorist, but you can use other theorists included in
# autora (for a list: https://autoresearch.github.io/autora/theorist/)

theorist = LinearRegression()

# To use the theorist on the state object, we wrap it with the on_state functionality and return a
# Delta object.
# Note: The if the input arguments of the theorist_on_state function are state-fields like
# experiment_data, variables, ... , then using this function on a state object will automatically
# use those state fields.
# The output of these functions is always a Delta object. The keyword argument in this case, tells
# the state object witch field to update.


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


# ** Experiment Runner ** #
# We will run our experiment on firebase and need credentials. You will find them here:
# (https://console.firebase.google.com/)
#   -> project -> project settings -> service accounts -> generate new private key
"""
firebase_credentials = {
  "type": "service_account",
  "project_id": "affective-stroop-ij",
  "private_key_id": "dc0d927e235dc90cbff9bf272de4f6048edc7ef7",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCxuvqVP9Yov7Le\now0VRhQjRRUrVAeozWJr4L2P80NYbYb10vDd4T0jq9iUdimEKPoHDcxjXmdKXVKX\nNqEbYIjLD81hx4AtRyHpMuvBVlesDOoq/jh1sKZF5/nFLeRMoF2l+05Jp4xT+lgT\nWi13HltA+z+MoghwpG+NUuMkvG2KeBWhzBcdOPKOizziHvXtRcCrbuS0WFrM2I9Q\neDibBjj2c9d8fvISrL7joDyYSrXiD3fkLhWvOnaX2VZNmNxEjeBAIZgFeY0IMHNI\nMUOTBeuYKcOXy8UVlil9r6QjQ2SNL8DAqVP02tRfb7KLkuM2kw8mLTdyI4j9bq+B\nh2+SgfjpAgMBAAECggEAJxXcGqt1wPNbdku8SlPlhNYh8CvkHVLBlMDRX1Fbm5xG\npFP15Y2toOUfVZLg0XAjgB6gNrAhngCE870vU/J4UtPQUnMJtKtigULSBuwijrMu\nYjMTHoqxHkmEK8FKZjGUNrbcAiWnHia9bNAVlI8n+VPnp1lQ6ti2sYkra75Ghr9A\nJqYQEhZEzP6Ew5QJ5W5psPVN68s205L8KHxUTyof5rJ0oXtwR21dn7Dns0aZNC1j\nM2vlUFlevu1OtemKWOW6PkcAG7YJQyZEqp1MEcQPSNNnUrxh+TTmArRG98Qr+Ux3\ndAZUxq/SXXrI/1UWyaKNQ45iIgZRfe/uDeUpjEK2WwKBgQD09kkMrt5j10vYkRg0\nIU7RNVUSVlxQxr53uHVYvXlSb4wspjstAAlFjjI19YQquyqGiwIHyuaXOKTT0kBH\nDRF3b74c+xHt3/MQ1DLRV102mhOM6QtQ2IEMvw0dnHoNRpAKtwfSOAOtcHbLoOMY\nFDtm/qs5cj0Ru4Tpu53Y4xnPnwKBgQC5vSetGJmUkFemkYHcCEMmyNRB8tJiYKhk\nJCXH+aSKgNwCaBDZcJJ+A1lzPZZEDZ3pyzj0Vh5xEj1n/EB7zw0E6NfRNgJAnR2V\nl8/3viFQAbT61HLfjF318yl0RmCwkVnmibOKA7y2PmTcrvxh9ZD8lfXckEhPO/2m\nFpFGcfHKdwKBgCd/OoIQBA2nHjSaGIP3RuUWq1Pclt/zW6ykEk+vpUs0by6X0vzL\nf7M2XUQ0BQ56xT2SVk55WbrSSElbIkASjFlsrAOyO87WCDQTcuosIvx11WcoAEoV\nzqEGgQ/gNhHD6d/AaUNtArLJam4kz9kHEhjmQSdvEwez65dgR/A7leLTAoGASn9R\n4J2TTAExBsAp48T0miPG7sSkGY1HPRM6jCDXHC5fECAJUg1S1HQndo9nK9iCU/mM\n2IOKJjP4qTKF8qKzSBrUJTeSY7Qg35VKdsWtutdUbZ60jHFSZomnQNnRgkBqXIsy\nldK16CrSbalJ1UiXZJgJSvQb7JvVOr6meijNoDUCgYEAja0BCJXcGPNNOyMfOq3D\nSk6TkDmLmOvjz0b8irCJh1N/ZMQ+AANm3Xgvlt6FU/mOkU4rI5bKTyJ9d7/cFpU8\nCML3fULaE+JrGumV77CKZpQ+DbKBgueFvliJwN264/kF2A5Qysen8CvwQmNFzlRT\nEpdciznkHpAWnatqMvYTBEw=\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-fbsvc@affective-stroop-ij.iam.gserviceaccount.com",
  "client_id": "109278083088643401587",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40affective-stroop-ij.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
"""
firebase_credentials = json.load(open("firebase_credentials.json"))

# simple experiment runner that runs the experiment on firebase
experiment_runner = firebase_runner(
    firebase_credentials=firebase_credentials,
    time_out=100,
    sleep_time=5)

# accuracy and mean RT are computed per phase
def compute_phase_metrics(df, exclude_practice=True):
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
 


# runner on state
@on_state()
def runner_on_state(conditions):

    js_codes = []
    for _, row in conditions.iterrows():
        length = int(row["length"])
        js_code =trial_sequence(length)
        js_codes.append(js_code)
    #these are the conditions we send to firebase (only the experiment code), more information than just length
    conditions_to_send = conditions.copy()
    conditions_to_send["experiment_code"] = js_codes

    # raw data: u.a. trials(enthält alle recordings vom participant wie acc, rt)
    raw_data = experiment_runner(conditions_to_send)               

    # parse and compute phase from raw data
    df = parse_raw_experiment_data(raw_data)
    phase_summary = compute_phase_metrics(df, exclude_practice=True)

    '''
    example phase summary: 
    phase	    accuracy	mean_rt	n_trials
    neutral1	0.85	    634.2	10
    emotional	0.78	    712.5	10
    neutral2	0.88	    645.1	10

    maybe experiemnt_data not necessary??
    '''

    return Delta(
        experiment_data=df,
        phase_summary=phase_summary
    )


# -> mit config, conditions??
# funciton für Leon
def wrapper_function():
    info= "Lieber Leon, wir hoffen du wirst hiermit glücklich:D LG"
    return runner_on_state, variables, info



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
