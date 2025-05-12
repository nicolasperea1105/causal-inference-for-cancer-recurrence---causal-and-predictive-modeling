# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:46:07 2025

@author: nicol
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import joblib
## Some steps can be generalized, and pipelines will be used for them, hence:
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

clinic_data = pd.read_csv("C:/Users/nicol/Downloads/brca_metabric_clinical_data.tsv",
                          sep = '\t')


        ## Functions used:
            
def fill_menopausal_state(df): ## Fills menopausal state based on age.

    ## According to WHO, women transition from pre to post menopausal states
    ## between the ages of 45 to 55, so 50 will be used as a cutoff for missing values.
    
    ## Then, if a woman is over 50 -> Post, under 50 -> Pre

    df.loc[(df['Inferred Menopausal State'].isna()) &
           (df['Age at Diagnosis'] >= 50),
           'Inferred Menopausal State'] = 'Post'
    
    df.loc[(df['Inferred Menopausal State'].isna()) &
           (df['Age at Diagnosis'] < 50),
           'Inferred Menopausal State'] = 'Pre'
    
    return df

def fill_radiotherapy(df): ## Fills YES if tumor size is more than median or positive lymphs>0
    
    df.loc[(df['Radio Therapy'].isna()) &
           ((df['Tumor Size'] > df['Tumor Size'].median())
            | (df['Lymph nodes examined positive'] > 0)),
           'Radio Therapy'] = 'YES'
    
    df.loc[(df['Radio Therapy'].isna()), 'Radio Therapy'] = 'NO'
    
    return df

def fill_tumorsize(df): ## Fills using segmented data based on radio therapy needs
    
    ## Median tumor size by radio therapy
    radio_yes_median = df[df['Radio Therapy'] == 
                          'YES']['Tumor Size'].median()
    radio_no_median = df[df['Radio Therapy'] == 
                         'NO']['Tumor Size'].median()

    df.loc[(df['Tumor Size'].isna()) &
           (df['Radio Therapy'] == 'YES'),
           'Tumor Size'] = radio_yes_median
    
    df.loc[(df['Tumor Size'].isna()) &
           (df['Radio Therapy'] == 'NO'),
           'Tumor Size'] = radio_no_median
    
    return df
   
    
def fill_positivelymphs(df): ## Fills using segmented data based on radio therapy needs
    
    ## Median amount of positive lymphs by radio therapy
    radio_yes_median = df[df['Radio Therapy'] == 
                          'YES']['Lymph nodes examined positive'].median()
    radio_no_median = df[df['Radio Therapy'] == 
                         'NO']['Lymph nodes examined positive'].median()
    
    df.loc[(df['Lymph nodes examined positive'].isna()) &
           (df['Radio Therapy'] == 'YES'),
           'Lymph nodes examined positive'] = radio_yes_median
    
    df.loc[(df['Lymph nodes examined positive'].isna()) &
           (df['Radio Therapy'] == 'NO'),
           'Lymph nodes examined positive'] = radio_no_median
    
    return df


def fill_hormonetherapy(df): ## Fills based on ER and PR values

    df.loc[(df['Hormone Therapy'].isna()) &
           ((df['ER Status'] == 'Positive') | 
            (df['PR Status'] == 'Positive')), 
           'Hormone Therapy'] = 'YES'
    
    df.loc[(df['Hormone Therapy'].isna()) & 
           ((df['ER Status'] == 'Negative') &
            (df['PR Status'] == 'Negative')),
           'Hormone Therapy'] = 'NO'
    
    df.loc[df['Hormone Therapy'].isna(),
           'Hormone Therapy'] = 'UNKOWN'
    
    return df
    
def fill_HER2(df): ## Fills based on Chemotherapy and Neoplasm Histologic Grade.
    
    df.loc[(df['HER2 Status'].isna()) &
           (df['Chemotherapy'] == 'YES') &
           (df['Neoplasm Histologic Grade'] == 3), 
           'HER2 Status'] = 'Positive'
    
    df.loc[df['HER2 Status'].isna(), 'HER2 Status'] = 'Negative'
    
    return df

def fill_ER(df): ## Fills based on Hormone therapy (Note: imputed based on imputed values)
    
    df.loc[(df['ER Status'].isna()) &
           (df['Hormone Therapy'] == 'YES'),
           'ER Status'] = 'Positive'
    
    df.loc[df['ER Status'].isna(), 'ER Status'] = 'Negative'
    
    return df
    
    
def fill_PR(df): ## Fills based on Hormone therapy (Note: imputed based on imputed values)
    
    df.loc[(df['PR Status'].isna()) &
           (df['Hormone Therapy'] == 'YES'),
           'PR Status'] = 'Positive'
    
    df.loc[df['PR Status'].isna(), 'PR Status'] = 'Negative'
    
    return df
    

def fill_chemotherapy(df): ## Fills based on HER2 (Note: imputed based on imputed values)

    df.loc[(df['Chemotherapy'].isna()) & 
           (df['HER2 Status'] == 'Positive'),
           'Chemotherapy'] = 'YES'
    
    df.loc[df['Chemotherapy'].isna(), 'Chemotherapy'] = 'NO'
    
    return df


def fill_neoplasm(df): ## Fills based on HER2 (Note: imputed based on imputed values)
    
    df.loc[(df['Neoplasm Histologic Grade'].isna()) &
           (df['HER2 Status'] == 'Positive'),
           'Neoplasm Histologic Grade'] = 3
    
    df.loc[(df['Neoplasm Histologic Grade'].isna() &
            (df['Chemotherapy'] == 'YES')),
           'Neoplasm Histologic Grade'] = 2
    
    df.loc[df['Neoplasm Histologic Grade'].isna(), 
           'Neoplasm Histologic Grade'] = 1
    
    return df
    
    
def fill_tumorstage(df): ## Fills based on tumor size (No stage 4 accounted for)
    
    df.loc[(df['Tumor Stage'].isna()) &
           (df['Tumor Size'] <= 20),
           'Tumor Stage'] = 1
    
    df.loc[(df['Tumor Stage'].isna()) &
           ((df['Tumor Size'] > 20) & 
            (df['Tumor Size'] <= 50)),
           'Tumor Stage'] = 2
    
    df.loc[(df['Tumor Stage'].isna()) &
           (df['Tumor Size'] > 50),
           'Tumor Stage'] = 3
    
    return df

def fill_surgery(df):
    
    df.loc[(df['Type of Breast Surgery'].isna()) &
           ((df['Tumor Size'] >= 30) |
            (df['Tumor Stage'] == 4)), 
           'Type of Breast Surgery'] = 'MASTECTOMY'
    
    df.loc[df['Type of Breast Surgery'].isna(),
           'Type of Breast Surgery'] = 'BREAST CONSERVING'
    
    return df
    
    
def fill_cellularity(df): ## Fills based on histologic grade (Note: imputed based on imputed values)
    
    df.loc[(df['Cellularity'].isna()) & 
           (df['Neoplasm Histologic Grade'] == 3),
           'Cellularity'] = 'High'

    df.loc[(df['Cellularity'].isna()) & 
           (df['Neoplasm Histologic Grade'] == 2),
           'Cellularity'] = 'Moderate'

    df.loc[(df['Cellularity'].isna()) & 
           (df['Neoplasm Histologic Grade'] == 1),
           'Cellularity'] = 'Low'
    
    return df


def binary_str_consistency(df, column, positive, negative):
    
    df.loc[df[column] == positive, column] = 'yes'
    df.loc[df[column] == negative, column] = 'no'
    
    return df


def reduce_memory(df): ## Reduces memory by casting columns of type object with
                       ## low cardinality into category type
    column_list = list(df.columns)
    
    for col in range(len(column_list)):
        
        if((df[column_list[col]].dtype == 'O') &
           (df[column_list[col]].value_counts().shape[0] < 20)):
            
            df[column_list[col]] = df[column_list[col]].astype('category')
            
    return df


def remove_by_IQR(df, column):
    
    Q1 = np.quantile(df[column], 0.25)
    Q3 = np.quantile(df[column], 0.75)
    limit = 1.5 * (Q3-Q1)
    
    df = df.loc[(df[column] > Q1-limit) &
                (df[column] < Q3+limit), :]
    
    return df

def remove_by_zscore(df, column, sds): ## standard deviations.
    
    mean = df[column].mean()
    sd = np.std(df[column])
    
    df = df.loc[(df[column] > mean-sds*sd) &
                (df[column] < mean+sds*sd), :]
    
    return df

def remove_outliers(df, column): ## If the data is not significantly skewed,
                                 ## kurtosis will define the sd cutoff for
                                 ## outlier removal through z-score.
                                 
    skewness = df[column].skew()
    kurtosis = df[column].kurtosis()
    
    if(abs(skewness) > 1.5):
        df = remove_by_IQR(df, column)
    else:
        if(kurtosis>=3):
            df = remove_by_zscore(df, column, 3)
        else:
            df = remove_by_zscore(df, column, 2)

    return df



    
        ## Pipeline usage
        
## Be aware that some steps in the pipeline are only adapted to this specific
## dataset and situation.

## All of the steps are created first and added to the pipeline at the end.

class MissingValuesImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X = fill_menopausal_state(X)
        X = fill_radiotherapy(X)
        X = fill_tumorsize(X)
        X = fill_positivelymphs(X)
        X = fill_hormonetherapy(X)
        X = fill_HER2(X)
        X = fill_ER(X)
        X = fill_PR(X)
        X = fill_chemotherapy(X)
        X = fill_neoplasm(X)
        X = fill_tumorstage(X)
        X = fill_surgery(X)
        X = fill_cellularity(X)
        
        return X
    
class BinaryConsistency(BaseEstimator, TransformerMixin):
    
    def __init__(self, column, positive, negative):
        self.column = column
        self.positive = positive
        self.negative = negative
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X = binary_str_consistency(X, self.column, self.positive, self.negative)
        
        return X
        
        
    
class ReduceMemory(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X = reduce_memory(X)
        
        return X
    
class RemoveOutliers(BaseEstimator, TransformerMixin):
    
    def __init__(self, column):
        self.column = column
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        
        X = X.copy()
        
        X = remove_outliers(X, self.column)
    
        return X
    
    
        ## Incorporating classes into pipeline objects. Explanation of why
        ## multiple pipelines in README.
        
missing_vals_pipe = Pipeline([
    ('missing_vals', MissingValuesImputer())])

def binary_consistency_pipe(column, positive, negative):
    return Pipeline([
        'binary_consistency', BinaryConsistency(column, positive, negative)])

reduce_memory_usage_pipe = Pipeline([
    'reduce_memory_usage', ReduceMemory()])

def remove_outliers_pipe(column):
    return Pipeline([
        'remove_outliers', RemoveOutliers(column)])

    
        ## Selecting relevant columns:
    
## The columns are selected based on medical importance to cancer recurrence.
## In this sense, columns like 'Patient ID' or 'Sample ID' will be removed while
## columns like 'Lymph nodes examined positive will remain in the data.
## Columns might be removed for other reasons, like 'Sex', since this is a
## breast cancer dataset.

clinic_data.columns

columns = [
  'Age at Diagnosis', 'Type of Breast Surgery', 'Cellularity',
  'Chemotherapy', 'ER Status', 'HER2 Status', 'Hormone Therapy',
  'Inferred Menopausal State', 'Lymph nodes examined positive',
  'Neoplasm Histologic Grade', 'PR Status', 'Radio Therapy',
  'Relapse Free Status', 'Tumor Size', 'Tumor Stage'
]

clinic_data = clinic_data.loc[:, columns]

## Selected columns:
    
print("Columns included:", clinic_data.columns.values)


        ## Handling missig values:
            
            
clinic_data.isna().sum()


    ## Age at diagnosis and inferred menopausal state:

clinic_data['Inferred Menopausal State'].value_counts() ## Only Pre or Post
clinic_data = fill_menopausal_state(clinic_data)

## Now there are 11 records with both age and menopausal state missing. Proceed by:
    
clinic_data['Age at Diagnosis'].fillna(clinic_data['Age at Diagnosis'].mean()
                                       , inplace=True)

## Now that the 11 missing values for age are filled, make a second pass for
## menopausal state.

clinic_data = fill_menopausal_state(clinic_data)
## This leaves no missing values for age or menopausal state.


    ## Radiotherapy, tumor size and lymph nodes
        
clinic_data = fill_radiotherapy(clinic_data) ## No missing values for radio therapy.

clinic_data = fill_tumorsize(clinic_data) ## No missing values for tumor size.

## The following line shows that tumor size and positive lymph nodes are not
## significantly correlated and therefore size will not be a factor when imputing for lymphs.
clinic_data[['Tumor Size', 'Lymph nodes examined positive']].corr() ## 0.269

clinic_data = fill_positivelymphs(clinic_data) ## No missing values for Lymph nodes examined positive


    ## ER, PR, HER2 and Hormone therapy
    
clinic_data = fill_hormonetherapy(clinic_data) ## No missing values for hormone therapy.
clinic_data = clinic_data.loc[clinic_data['Hormone Therapy'] != 'UNKOWN', :]
## Drop values where hormone therapy is unkown.


clinic_data = fill_HER2(clinic_data) ## No missing values for HER2.

## Due to the following 2 categories being filled using already imputed values,
## they should be taken into account as biased and not used in real medical scenarios.

clinic_data = fill_ER(clinic_data) ## No missing values for ER.
clinic_data = fill_PR(clinic_data) ## No missing values for PR.

    ## Chemotherapy and Neoplasm Histologic Grade
    
## Due to the following 2 categories being filled using already imputed values,
## they should be taken into account as biased and not used in real medical scenarios.

clinic_data = fill_chemotherapy(clinic_data) ## No missing values for chemotherapy.
clinic_data = fill_neoplasm(clinic_data) ## No missing values for Neoplasm Histologic Grade.


    ## Tumor stage
    
## Note that this doesnt account for stage 4 tumors and this result could be
## significant for recurrence.
clinic_data = fill_tumorstage(clinic_data) ## No missing values for tumor stage.


    ## Type of Breast Surgery
    
## This imputation is also not medically sufficient, since it only accounts for
## tumor size and stage in order to define the type of surgery needed.

clinic_data = fill_surgery(clinic_data) ## No missing values for Type of Breast Surgery.


    ## Cellularity
    
## Due to the following category beiyg filled using already imputed values,
## it should be taken into account as biased and not used in real medical scenarios.

clinic_data = fill_cellularity(clinic_data) ## No missing values for Cellularity.


    ## Relapse Free Status.
    
## Since this is the target variable of the study, the rows with missing values
## will not be imputed but dropped.

clinic_data = clinic_data.dropna(subset=['Relapse Free Status'])


## Total missing values after consideration:
    
print("Final missing values:", clinic_data.isna().sum().sum()) ## 0


        ## Categorical consistency
        
## Binary categorical columns

clinic_data = binary_str_consistency(clinic_data, 
                                     'Chemotherapy', 
                                     'YES', 'NO')

clinic_data = binary_str_consistency(clinic_data, 
                                     'ER Status', 
                                     'Positive', 'Negative')

clinic_data = binary_str_consistency(clinic_data, 
                                     'HER2 Status', 
                                     'Positive', 'Negative')

clinic_data = binary_str_consistency(clinic_data, 
                                     'Hormone Therapy', 
                                     'YES', 'NO')

clinic_data = binary_str_consistency(clinic_data, 
                                     'PR Status', 
                                     'Positive', 'Negative')

clinic_data = binary_str_consistency(clinic_data, 
                                     'Radio Therapy', 
                                     'YES', 'NO')

        ## Handling memory usage:
            
unoptimized_memory = clinic_data.memory_usage(deep=True).sum()
print("Unoptimized memory usage in bytes:", unoptimized_memory)

clinic_data = reduce_memory(clinic_data)

optimized_memory = clinic_data.memory_usage(deep=True).sum()
print("Optimized memory usage in bytes:", optimized_memory)

memory_reduction_percent = (1 - optimized_memory/unoptimized_memory)*100
memory_reduction_total = unoptimized_memory-optimized_memory

print("Memory usage reduced from", unoptimized_memory, "bytes to", optimized_memory,
      'bytes \n', "This represents a reduction of", memory_reduction_total, "bytes or",
      memory_reduction_percent, "percent.")

## Memory usage reduced from 1504106 bytes to 146662 bytes 
## This represents a reduction of 1357444 bytes or 90.24922445625508 percent.



        ## Handling outliers
        
clinic_data.dtypes
## Ouliers from the followig categories will be removed will be removed:
## Age, Positive lymph nodes, Tumor size. The other numerical columns do not
## need this, as they measure a grade of stage bound from 1 to 3 and 1 to 4
## in this specific data.

length_before = clinic_data.shape[0]

clinic_data = remove_outliers(clinic_data, 'Age at Diagnosis')
clinic_data = remove_outliers(clinic_data, 'Lymph nodes examined positive')
clinic_data = remove_outliers(clinic_data, 'Tumor Size')

length_after = clinic_data.shape[0]

print("Outliers removed:" , length_before-length_after) ## 439


        ## Duplicates
        
## In this case, and since ID is not present, it is possible that 2 people had
## the same conditions and treatments, so duplicated data is not a concern.



## This concludes the data cleaning stage.
## Saving pkl files for future usage:
    
clinic_data.to_pickle("clinic_data_clean.pkl")

joblib.dump(missing_vals_pipe, 'missing_vals_pipe')
joblib.dump(binary_consistency_pipe(None, None, None), 
            'binary_consistency_pipe_initializer')
joblib.dump(reduce_memory_usage_pipe, 'reduce_memory_usage_pipe')
joblib.dump(remove_outliers_pipe(None), 'remove_outliers_pipe_initializer')