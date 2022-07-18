import pandas as pd

def get_features(df, corr_thresh=.01, corr_df=False):
    """
    Select the features from the columns by applying the cross correlation method.
    :param df: Pandas dataframe
        The data
    :param corr_thresh: float
        Minimum correlation threshold between the target and other columns
    :param corr_df: Pandas dataframe
        Correlation values between the target and other columns
    :return:
        features: list
        correlation table: dataframe (optional)
    """
    corr_list = []  # to keep the correlations with price
    for col in df.columns:
        corr_list.append(round(df["TARGET"].corr(df[col]), 2))
    df_corr = pd.DataFrame(data=zip(df.columns.tolist(), corr_list),
                           columns=["col_name", "corr"]) \
        .sort_values("corr", ascending=False) \
        .reset_index(drop=True)
    df_corr = df_corr[abs(df_corr["corr"]) > corr_thresh][1:].reset_index(drop=True)
    features = df_corr["col_name"].tolist()
    if not corr_df:
        return features
    return features, df_corr