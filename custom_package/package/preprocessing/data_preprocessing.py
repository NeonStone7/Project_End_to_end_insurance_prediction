from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
import numpy as np
import pandas as pd

def create_pipeline(num_pipeline:Pipeline, num_columns:list, cat_pipeline:Pipeline, cat_columns:list):
    """This function creates a custom pipeline for data transformation for numerical and categorical columns
    Inputs:
    num_pipeline: Sklearn pipeline for numerical transformations
    num_columns: Quantitative columns, type -->  list
    cat_pipeline: Sklearn pipeline for categorical transformations
    cat_columns: Qualitative data, type -->  list
    
    
    """
    
    pipeline  = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_columns),
        ('cat_pipeline', cat_pipeline, cat_columns),
        
    ], remainder = 'passthrough', verbose_feature_names_out = False)
    
    return pipeline


def winsorize(dataframe):
    
    "This function attempt to reduce outliers by replacing values the 95th and 5th Percentile with the percentile value"
    
    for column in dataframe:
        
        q95 = dataframe[column].quantile(0.95)
        
        q5 = dataframe[column].quantile(0.05)
        
        dataframe[column] = np.where(dataframe[column] > q95, q95,
                                    np.where(dataframe[column] < q5, q5, dataframe[column]))
        
    return dataframe

def percentile_imputer(dataframe):
    
    """This function replaces null values with the values at the 95th percentile"""
    
    for column in dataframe:
        
        percentile_95 = dataframe[column].quantile(0.95)
        
        dataframe[column] = dataframe[column].fillna(percentile_95)
        
    return dataframe

def random_sample_imputer(dataframe):
    
    """This function replaces null values with random samples drawn from the dataset"""
    
    for column in dataframe:
        
        # count the number of nan values in the column
        nans = dataframe[column].isnull().sum()
        
        # take a random-sample of not-null values
        sample = dataframe[column].dropna().sample(nans, replace=True)
        
        # match the index of the sample with that of the dataframe
        sample.index = dataframe.loc[dataframe[column].isnull(), column].index
        
        # set the sample to the null values
        dataframe.loc[dataframe[column].isnull(), column] = sample
        
    return dataframe

def count_encoder(dataframe):
    
    """This function encodes numerical columns by replacing each value with the frequency of the value"""
    
    for column in dataframe:
        
        counts = dataframe[column].value_counts(normalize = True).to_dict()
        
        dataframe[column] = dataframe[column].map(counts)
        
        
    return dataframe