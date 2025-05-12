# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:02:06 2025

@author: nicol
"""


import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

clinic_data = pd.read_pickle("C:/Users/nicol/OneDrive/Desktop/Predicting cancer recurrence through causal inference/clinic_data_clean.pkl")


    ## Functions used:
        
        
def encode(df, column, method, pos=None, neg=None, order=None):
    
    if method == 'binary':
        df[column] = df[column].map({pos:1, neg:0})
        
    elif method == 'ordinal':
        encoder = OrdinalEncoder(categories=[order])
        df[column] = encoder.fit_transform(df[[column]])
        
    elif method == 'nominal':
        encoder = OneHotEncoder(sparse_output=False, drop=None)
        encoded = encoder.fit_transform(df[[column]])
        col_names = encoder.get_feature_names_out([column])
        df = df.drop(columns=[column])
        df[col_names] = encoded
    else:
        raise ValueError("method should be binary, ordinal or nominal.")
    
    return df



clinic_data.dtypes
        ## Encoding


    ## Binary


clinic_data = encode(clinic_data, 'Chemotherapy', 'binary',
                     'yes', 'no')
clinic_data = encode(clinic_data, 'ER Status', 'binary',
                     'yes', 'no')
clinic_data = encode(clinic_data, 'HER2 Status', 'binary',
                     'yes', 'no')
clinic_data = encode(clinic_data, 'Hormone Therapy', 'binary',
                     'yes', 'no')
clinic_data = encode(clinic_data, 'PR Status', 'binary',
                     'yes', 'no')
clinic_data = encode(clinic_data, 'Radio Therapy', 'binary',
                     'yes', 'no')

## The column addressing recurrence needs a name change to be representative
## of the outcome after encoding, hence:
    
clinic_data.rename(columns={'Relapse Free Status':'Recurrence'}, inplace=True)
clinic_data = encode(clinic_data, 'Recurrence', 'binary',
                     '1:Recurred', '0:Not Recurred')

## According to the EDA, categorical variables do not act with monotonicity
## in realtion to the outcome variable, and therefore these will also be
## encoded as nominal variables.


    ## Nominal
    
## In this case, OneHotEncoded columns should all be kept. For example,
## a 0 in Mastectomy could imply no surgery, when it in fact means a breast
## conserving surgery. Even though keeping both columns will introduce
## multicollinearity, it is important for the causal section that they remain
## included.
    

clinic_data = encode(clinic_data, 'Type of Breast Surgery', 'nominal')
clinic_data = encode(clinic_data, 'Cellularity', 'nominal')
clinic_data = encode(clinic_data, 'Inferred Menopausal State', 'nominal')


## The variables encoded with 'binary' still are set to type category,
## which is not optimal for modeling. These, as well as float64 category
## covariates will be transformed to float32 to improve functionality and
## memory usage.

print(clinic_data.dtypes) ## This shows every series needs to be converted.


clinic_data.columns

for col in list(clinic_data.columns):
    clinic_data[col] = clinic_data[col].astype('float32')
    
    
print(clinic_data.dtypes) ## All columns are now of type float32.


    ## Replace space with _
    
new_names = []

for col in list(clinic_data.columns):
    name = col.replace(' ', '_')
    new_names.append(name)
    
clinic_data.columns = new_names
    
print(clinic_data.columns) ## Now every space is replaced by a _



        ## Export resulting file

clinic_data.to_pickle("clinic_data_engineered.pkl")    
