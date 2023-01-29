# imports to run my functions
import pandas as pd
import os
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

# personal creds
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

#--------------------------------

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
    

#--------------------------------
    
def get_mall_data():
    
    """returns a dataframe from SQL of all 2017 properties that are single family residential"""

    sql ="""
         select * from customers
        """
    return pd.read_sql(sql, get_connection("mall_customers"))
    
#--------------------------------
    
    
def wrangle_mall():
    
    """
   Acquires mall data from sql ace database
    """

    # create a variable name
    filename = "mall.csv"

    # searching for that variable name
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
    
        # perform other functions to make a new data acquistion
        df = get_mall_data()
        
        df.to_csv('mall.csv')
        
    df = df.drop('Unnamed: 0', axis=1)    
    return df

#--------------------------------  

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    
    print(train.shape , validate.shape, test.shape)

          
    return train, validate, test

#-------------------------------

def get_dummies(df,cols):
    
    df[cols + '_dummy'] = pd.get_dummies(df[cols],drop_first= True)
    
    df.drop(columns= cols)
    
    return df
    
#-------------------------------

def scaled_data(x_train,x_validate,x_test,num_cols,return_scaler = False):

    ''' a function to scale my data appropriately ''' 
    
    # intializing scaler
    scaler = MinMaxScaler()
    
    # fit scaler
    scaler.fit(x_train[num_cols])
    
    # creating new scaled dataframes
    x_train_s = scaler.transform(x_train[num_cols])
    x_validate_s = scaler.transform(x_validate[num_cols])
    x_test_s = scaler.transform(x_test[num_cols])

    # making a copy of train to hold scaled version
    x_train_scaled = x_train.copy()
    x_validate_scaled = x_validate.copy()
    x_test_scaled = x_test.copy()

    x_train_scaled[num_cols] = x_train_s
    x_validate_scaled[num_cols] = x_validate_s
    x_test_scaled[num_cols] = x_test_s

    if return_scaler:
        return scaler, x_train_scaled, x_validate_scaled, x_test_scaled
    else:
        return x_train_scaled, x_validate_scaled, x_test_scaled
    
#-------------------------------
    
def rename_cols(df):
   
    ''' a function to rename columns and make them easier to read '''
    
    # renaming method performed 
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                            'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet':'sq_feet', 
                            'taxvaluedollarcnt':'tax_value',
                            'yearbuilt':'year_built',
                            'taxamount':'tax_amount'})
    return df

#-------------------------------
    

def change_zillow(df):
    
    ''' a function to change data types of my columns and map names to fips'''
    
    # replacing nulls with zero
    df['poolcnt'] = df['poolcnt'].replace(np.nan, 0)
    
    df['unitcnt'] = df['unitcnt'].replace(np.nan, 0)
    
    df['heatingorsystemtypeid'] = df['heatingorsystemtypeid'].replace(np.nan, 0)
    
    df['garagetotalsqft'] = df['garagetotalsqft'].replace(np.nan, 0)
    
    df['fullbathcnt'] = df['fullbathcnt'].replace(np.nan, 0)
   
    df['fireplacecnt'] = df['fireplacecnt'].replace(np.nan, 0)
    
    df['basementsqft'] = df['basementsqft'].replace(np.nan, 0)
    
    
    # mapping fips code to the county
    df['fips'] = df.fips.map({ 06037.0: 'Los Angeles', 06059.0: 'Orange', 06111.0: 'Ventura'})
    
    
    return df


#----------------------
def clean_zillow(df):
    
    '''
    takes data frame and changes datatypes and renames columnns, returns dataframe
    '''
    # calls other functions
    df = change_zillow(df)
    
    df = handle_outliers(df)
    
    df = rename_cols(df)
    
    df_d = pd.get_dummies(df,columns= ['bedrooms','bathrooms','assessmentyear','poolcnt','unitcnt','heatingorsystemtypeid','fullbathcnt','fireplacecnt'],drop_first = True)
        
    # save df to csv
    df.to_csv("zillow.csv", index=False)

    return df, df_d

#----------------------

def handle_outliers(df):
    
    '''handle outliers that do not represent properties likely for 99% of buyers and zillow visitors'''
    
    # this series of steps is how outliers were determined and removed
    
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 2_000_000]

    df = df[df.calculatedfinishedsquarefeet < 10000]
    
    df = df[df.yearbuilt > 1850]

    return df

#----------------------

def x_and_y(train,validate,test,target):
    
    """
    splits train, validate, and target into x and y versions
    """

    x_train = train.drop(columns= target)
    y_train = train[target]

    x_validate = validate.drop(columns= target)
    y_validate = validate[target]

    x_test = test.drop(columns= target)
    y_test = test[target]

    return x_train,y_train,x_validate,y_validate,x_test, y_test


#----------------------

def change_dtype(df):
    
    float_cols = ['regionidzip','basementsqft', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'fireplacecnt' ,'fullbathcnt', 'garagetotalsqft', 'heatingorsystemtypeid', 'latitude', 'longitude', 'poolcnt', 'propertylandusetypeid', 'regionidcity','roomcnt','unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt','taxvaluedollarcnt', 'assessmentyear','landtaxvaluedollarcnt']
    
    for col in df.columns:
        if col in float_cols:
            df[col] = df[col].astype(int)
        
    return df