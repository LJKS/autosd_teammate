from autora.theorist.bms import BMSRegressor
import pandas as pd
from autora.state import on_state
import ray
def has_no_data(df):
    num_rows = df.shape[0]
    return num_rows == 0

def get_empty_theorist():
    #empty theorist has fit and predict methods that do nothing, predict returns zeros
    class EmptyTheorist:
        def fit(self, x, y):
            return self
        def predict(self, x):
            import numpy as np
            return np.zeros((x.shape[0], 1))
        def __str__(self):
            return "EmptyTheorist"
    return EmptyTheorist()
def join_dicts(dicts):
    """
    Join a list of dictionaries into a single dictionary.
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result

def remove_NaN_rows(df_x, df_y):
    """
    Remove rows with NaN values from both dataframes, ensuring alignment.
    """
    combined = pd.concat([df_x, df_y], axis=1)
    cleaned = combined.dropna()
    cleaned_x = cleaned[df_x.columns]
    cleaned_y = cleaned[df_y.columns]
    return cleaned_x, cleaned_y

def get_bms_theorist(epochs=10):
    @on_state(output=['models'])
    def bms_theorist(experiment_data, variables, preprocessing_function):
        independent_var_names = [var.name for var in variables.independent_variables]
        covariate_var_names = [var.name for var in variables.covariates]
        dependent_var_names = [var.name for var in variables.dependent_variables]
        x = experiment_data[independent_var_names + covariate_var_names]
        print(f'x pre preprocessing: {x}')
        x = preprocessing_function(x)
        print(f'x post preprocessing: {x}')
        y = experiment_data[dependent_var_names]
        #split y into seperate columns, as BMSRegressor expects a 1D array for y, then fit (parallel via ray) a separate BMSRegressor for each dependent variable
        if len(dependent_var_names) == 1:
            bms = BMSRegressor(epochs=epochs)
            y, x = remove_NaN_rows(x, y)
            bms.fit(x, y)
            return [bms]
        else:
            #check if ray is initialized, if not initialize it
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            columns = {col: y[[col]] for col in dependent_var_names}
            @ray.remote
            def fit_bms(col, x, col_data_y):
                print(f'Fitting BMS for column: {col}, with shapes x: {x.shape}, y: {col_data_y.shape}')
                print(f'Column {col} has input data {x}')
                #if col_data_y has no data, return an empty theorist
                bms = BMSRegressor(epochs=epochs)
                x, col_data = remove_NaN_rows(x, col_data_y)
                if has_no_data(col_data):
                    print(f'WARNING: Fitting theorist on empty data, returning empty theorist')
                    return {col: get_empty_theorist()}
                bms.fit(x, col_data)
                return {col: bms}
            #fit
            futures = [fit_bms.remote(col, x, col_data) for col, col_data in columns.items()]
            #combine list of bms dicts into a single dict
            results = ray.get(futures)
            bms_dict = join_dicts(results)
            return [bms_dict]
    return bms_theorist

def basic_affective_stroop_theorist_preprocessing(df):
    """
    The phase_summary dataframes look something like the following example:
                phase  accuracy  mean_rt  mean_rt_correct  mean_rt_incorrect  n_trials    valence
            0  emotional       1.0   671.75           671.75                NaN         4  emotional
            1   neutral1       1.0  1452.00          1452.00                NaN         4    neutral
            2   neutral2       1.0   524.00           524.00                NaN         4    neutral

    Notice the columns for valence and phase are categorical. The preprocessing replaces them with respective pseudo-one-hot encodings, i.e. phase_is_emotional, phase_is_neutral1, phase_is_neutral2, valence_is_emotional, valence_is_neutral.
    :param df: a dataframe with the phase summary data
    :return: the preprocessed dataframe
    """
    df = df.copy()
    #one-hot encode the categorical variables phase and valence
    df = pd.get_dummies(df, columns=['phase', 'valence'], prefix=['phase_is', 'valence_is'])
    #convert boolean columns to float
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(float)
    return df