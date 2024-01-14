# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def get_label_encoding(app_train,app_test):
    le = LabelEncoder()
    le_count = 0
    for col in app_train:
        if app_train[col].dtype == 'object':
            if len(list(app_train[col].unique())) <= 2:
                le.fit(app_train[col])
                app_train[col] = le.transform(app_train[col])
                app_test[col] = le.transform(app_test[col])
                le_count += 1
    return app_train,app_test

def main():

    app_train = pd.read_csv('../../data/application_train.csv')
    print('Training data shape: ', app_train.shape)
    app_train.head()
    app_test = pd.read_csv('../data/application_test.csv')
    print('Testing data shape: ', app_test.shape)
    app_test.head()
    # Missing values statistics
    missing_values = missing_values_table(app_train)
    # one-hot encoding of categorical variables
    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)
    app_train.fillna(-999, inplace = True)
    app_test.fillna(-999, inplace = True)


    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)


    rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
    rf.fit(app_train.drop(['SK_ID_CURR', 'TARGET'],axis=1), app_train.TARGET)
    features = app_train.drop(['SK_ID_CURR', 'TARGET'],axis=1).columns.values


    from econml.solutions.causal_analysis import CausalAnalysis
    ca = CausalAnalysis(top_features,categorical,heterogeneity_inds=None,classification=True,nuisance_models="automl",heterogeneity_model="forest",n_jobs=-1,random_state=123,)
    ca.fit(x_train, y_train.values)

