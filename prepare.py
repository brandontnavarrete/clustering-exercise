import pandas as pd 
import numpy as np 


def handle_missing_val(df, column_want , row_want):
    
    """
    This function takes in a dataframe, a column threshold, and a row threshold,
    and removes columns and rows based on the percentage of missing values.
    Columns are removed if the proportion of missing values is greater than the column threshold.
    Rows are removed if the proportion of missing values is greater than the row threshold.

    Parameters:
    dataframe
    
    column_want (float): a number between 0 and 1 representing the proportion of non-missing values required for each column
    row_want (float): a number between 0 and 1 representing the proportion of non-missing values required for each row

    Returns:
    df (pandas DataFrame): the cleaned dataframe with columns and rows removed as indicated
    """
    # Calculate missing values for each column
    missing_cols = df.isnull().sum() / len(df)
    # Identify columns that have missing values greater than the column threshold
    drop_cols = missing_cols[missing_cols > (1 - column_want)].index
    # Drop the columns
    df = df.drop(drop_cols, axis=1)

    # Calculate missing values for each row
    missing_rows = df.isnull().sum(axis=1) / df.shape[1]
    # Identify rows that have missing values greater than the row threshold
    drop_rows = missing_rows[missing_rows > (1 - row_want)].index
    # Drop the rows
    df = df.drop(drop_rows)
    return df