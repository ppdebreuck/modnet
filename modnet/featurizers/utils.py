import numpy as np

__all__ = ("clean_df",)


def clean_df(df, drop_allnan: bool = True):
    """Cleans dataframe by dropping missing values, replacing NaN's and infinities
    and selecting only columns containing numerical data.

    Args:
        df (pd.DataFrame): the dataframe to clean.
        drop_allnan: if True, clean_df will remove features that are fully NaNs.

    Returns:
        pandas.DataFrame: the cleaned dataframe.

    """

    df = df.select_dtypes(include="number")
    if drop_allnan:
        df = df.dropna(axis=1, how="all")
    df = df.replace([np.inf, -np.inf, np.nan], np.nan)

    return df
