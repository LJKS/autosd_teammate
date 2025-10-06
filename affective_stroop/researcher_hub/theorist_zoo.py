from autora.theorist.bms import BMSRegressor

@on_state(output=['models'])
def bms_theorist(experiment_data, variables, preprocessing_function, epochs=10):
    bms = BMSRegressor(epochs=10)
    independent_var_names = [var.name for var in variables.independent_variables]
    covariate_var_names = [var.name for var in variables.covariates]
    dependent_var_names = [var.name for var in variables.dependent_variables]
    x = experiment_data[independent_var_names + covariate_var_names]
    x = preprocessing_function(x)
    y = experiment_data[dependent_var_names]
    bms.fit(x, y)
    return [bms]

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
    return df