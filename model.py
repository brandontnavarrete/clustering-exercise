import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

import acquire as ac
import prepare as pr
import wrangle as wr




def plot_histogram(df):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    df = df.select_dtypes(include=numerics)
    
    for column in df.columns:
        # plt histogram
        plt.hist(df[column])
        # name title and labels
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()