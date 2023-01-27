# imports to run my functions
import pandas as pd
import os
import numpy as np
import env

# setting connectiong to sequel server using env

def get_connection(db, user=env.username, host=env.host, password=env.password):
    
    ''' a function to handle my sql ace creds'''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#----------------------------------------------
 
# acquiring zillow data using a get_connection

def get_zillow_data():
    
    """returns a dataframe from SQL of all 2017 properties that are single family residential"""

    sql = """
    select *
    from (select distinct parcelid, logerror, transactiondate from predictions_2017)
    as dups 
    join properties_2017 using (parcelid)
    """
    return pd.read_sql(sql, get_connection("zillow"))

#--------------------------------

def wrangle_zillow():
    
    """
   Acquires zillow data and uses the clean function to call other functions and returns a clean data        frame with new names,  dropped nulls, new data types.
    """

    # create a variable name
    filename = "zillow2017.csv"

    # searching for that variable name
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
    
        
        # perform other functions to make a new data acquistion
        df = get_zillow_data()
        
        df.to_csv('zillow2017.csv')
        
    df = df.drop('Unnamed: 0', axis=1)    
    return df
#--------------------------------

def missing_values(df):
    
    """
    This function takes in a dataframe and returns a new dataframe with 
    information about missing values in the original dataframe. 
    Each row in the returned dataframe corresponds to an attribute in the original dataframe, 
    the first column is the number of missing values for that attribute, 
    and the second column is the percentage of missing values for that attribute.

    Parameters: 
    Dataframe

    Returns: 
    Dataframe: a dataframe with information about missing values in the original dataframe named null_df
    """
    
    # the sum of values per each row  #reset_index( ) creates a dataframe while resetting the index
    null_df = df.isnull().sum().reset_index()
    
    # creating column names for data frame
    null_df.columns = ['Attribute', 'missing_count']
    
    # dividing the missing count by length of data frame then multiplying it by 100 for the missing percentage
    null_df['missing_perc'] = (null_df['missing_count'] / len(df)) * 100
    
    #return the dataframe
    return null_df
    