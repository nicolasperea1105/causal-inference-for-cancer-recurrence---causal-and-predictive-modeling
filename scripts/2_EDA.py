# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 23:18:25 2025

@author: nicol
"""


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


clinic_data = pd.read_pickle("myfilepath")

clinic_data.dtypes

        ## EDA
        
        
    ## Plot settings:
        
sns.set_style('darkgrid')
sns.set_context('talk')
sns.set_palette('deep')
        

## Correlation Tumor size - Tumor stage

print(clinic_data[['Tumor Size', 'Tumor Stage']].corr()) ## 0.537

## Correlation Positive lymph nodes - Histological grade

print(clinic_data[['Lymph nodes examined positive',
                   'Neoplasm Histologic Grade']].corr()) ## 0.12



    ## Outcome directed EDA: what seems to be related with recurrence?
    
## How many patiens did and did not recurr?
## The following dataset is imbalanced due to:
    
recurring = clinic_data.loc[clinic_data['Relapse Free Status'] == '1:Recurred', :]
non_recurring = clinic_data.loc[clinic_data['Relapse Free Status'] == '0:Not Recurred', :]

print("Recurred:", recurring.shape[0]) ## 659

print("Did not recur:", non_recurring.shape[0]) ## 1192
    
## Is tumor aggressiveness (histologic grade) related to recurrence?


plt.figure(figsize=(8, 5))
sns.countplot(data = clinic_data, x='Relapse Free Status',
            hue='Neoplasm Histologic Grade')
plt.title('Recurrence by histologic grade.', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## Although higher histologic grades are common amongst recurring patients, they are
## similarly prevalent in non-recurring patients. This points at this variable not
## being a strong predictor alone for recurrence.


## Is cancer spread (positive lymph nodes) related to recurrence?

plt.figure(figsize=(8, 5))
sns.countplot(data = clinic_data, x='Relapse Free Status',
            hue='Lymph nodes examined positive')
plt.title('Recurrence by positive lymph nodes.', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## This shows a similar behavior as tumor aggressiveness, trends on
## recurrence outcome across positive lymph nodes tend to maintain. The only
## significant difference is that the ratio of low amount of positive lymph
## nodes is larger for not recurring patients, referencing some signal regarding
## its correlation with recurrence.


## Is menopausal state related to recurrence?

plt.figure(figsize=(8, 5))
sns.countplot(data = clinic_data, x='Relapse Free Status',
            hue='Inferred Menopausal State')
plt.title('Recurrence by menopausal state.', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## Similar situation as before. In this case, pre menopausal state patient count
## is approximately equal for either outcome of recurrence, which paired with 
## non-recurrent patients being the majority of the data, means that patients
## with pre menopausal state are more prone to recurring.

## The data shows that the proportion of recurring pre menopausal patients
## is rougly double the proportion of non-recurring pre menopausal patients.
## This points at pre menopausal state being slightly related to recurrence.


## Is age related to recurrence?

plt.figure(figsize=(8, 5))
sns.histplot(data = clinic_data, x='Age at Diagnosis',
             hue = 'Relapse Free Status', multiple='stack')
plt.title('Recurring Patients by Age', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## This plot does not show an age range where recurrence is more likely to
## happen, only small fluctuations in proportion of recurring vs non-recurring.


## Is the type of surgery related to recurrence?

plt.figure(figsize=(8, 5))
sns.countplot(data = clinic_data, x='Relapse Free Status',
            hue='Type of Breast Surgery')
plt.title('Recurrence by surgery undergone.', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## No significant relation observed.


## Is cellularity related to recurrence?

plt.figure(figsize=(8, 5))
sns.countplot(data = clinic_data, x='Relapse Free Status',
            hue='Cellularity')
plt.title('Recurrence by Cellularity.', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## No relation observed.


## Is hormone therapy related to recurrence?

plt.figure(figsize=(8, 5))
sns.countplot(data = clinic_data, x='Relapse Free Status',
            hue='Hormone Therapy')
plt.title('Recurrence by Hormone Therapy.', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## There is a slightly higher proportion of patients that did not undergo 
## hormone therapy amongst recurring patients than amongst non-recurring.
## The use of hormone therapy can be related to recurrence.


## Is radio therapy related to recurrence?

plt.figure(figsize=(8, 5))
sns.countplot(data = clinic_data, x='Relapse Free Status',
            hue='Radio Therapy')
plt.title('Recurrence by Radio Therapy.', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## No significant relationship observed.

## Is the need for chemotherapy related to recurrence?

plt.figure(figsize=(8, 5))
sns.countplot(data = clinic_data, x='Relapse Free Status',
            hue='Chemotherapy')
plt.title('Recurrence by Chemotherapy.', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## Chemotherapy seems to be related with non-recurrence. This result appears 
## reliable, but it could be because of chemotherapy being effective or
## because patients that do not need chemotherapy naturally are less likely
## to recur.




        ## Treatment directed EDA: what seems to be related to having 
        ## chemotherapy?
        
chemotherapy_yes = clinic_data.loc[clinic_data['Chemotherapy'] == 'yes', :]
chemotherapy_no = clinic_data.loc[clinic_data['Chemotherapy'] == 'no', :]
        
## This data is also unbalanced. There is roughly 6 times the amount of patients
## that did not undergo chemotherapy than the ones that did.


## Is tumor stage related to chemotherapy?

plt.figure(figsize=(8, 5))
sns.countplot(data = clinic_data, x='Chemotherapy',
            hue='Tumor Stage')
plt.title('Chemotherapy Patients by Tumor Stage', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## As opposed to patients that did not undergo chemotherapy, the amount of 
## patients that did surges significantly from stage 2 to stage 3 tumors.
## In other words, the amount of patients that did not undergo chemotherapy
## are scattered across tumor stage, but the patients that did need chemotherapy
## are mostly the ones with stage 3 tumors.
## This points at tumor stage being significant when defining the need for
## chemotherapy.



## Is tumor size related to chemotherapy?

plt.figure(figsize=(8, 5))
sns.histplot(data = clinic_data, x='Tumor Size',
             hue = 'Chemotherapy', multiple='stack')
plt.title('Chemotherapy by tumor size', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## Tumor size is not shown to be an important feature when defining the need
## for chemotherapy, but sizes of around 24mm undergo chemotherapy more often.


## Is cancer spread (positive lymph nodes) related to chemotherapy?


plt.figure(figsize=(8, 5))
sns.countplot(data=clinic_data, x='Lymph nodes examined positive', 
              hue='Chemotherapy')
plt.title('Chemotherapy by cancer spread', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## Again, even though most patients across all amounts of positive lymph nodes
## do not undergo chemotherapy, the proportion of patients that do increases
## rapidly with positive lymph nodes. This also shows to be a strong indicator 
## for needing chemotherapy.


## Is tumor aggreswiveness (histologic grade) related to chemotherapy?


plt.figure(figsize=(8, 5))
sns.countplot(data=clinic_data, x='Neoplasm Histologic Grade', 
              hue='Chemotherapy')
plt.title('Chemotherapy by tumor aggresiveness', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()


## Same trend. Despite there being more patients with no chemotherapy across
## all histologic grades, the proportion of patients with chemotherapy increases
## significantly with this measure.


## Is cellularity related to chemotherapy?

plt.figure(figsize=(8, 5))
sns.countplot(data=clinic_data, x='Cellularity', 
              hue='Chemotherapy')
plt.title('Chemotherapy by Cellularity', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## Cellularity does not seem correlated to needing chemotherapy.



## Is HER2 status related to chemotherapy?

plt.figure(figsize=(8, 5))
sns.countplot(data=clinic_data, x='HER2 Status', 
              hue='Chemotherapy')
plt.title('Chemotherapy by HER2 Status', weight='bold', fontsize=16)
plt.tight_layout()
plt.show()

## Even though the proportion of patients with HER2 positive status is greater
## among the patients that underwent chemotherapy, the majority of patients 
## with this treatment still have a negative HER2 status. In different terms,
## chemotherapy can be applied for every HER2 status, but being HER2 positive
## increases the probability of having chemotherapy.

## This EDA shows that each available covariate on its own is not enough to
## predict recurrence, but that these can indicate if a patient will need 
## chemotherapy - or be assigned to the treatment for causal purposes - which
## is in turn somewhat related with recurrence.

## In this case, even if the covariates do not show a direct effect on the
## causal outcome, through domain knowledge it is clear that they still are 
## important for it. This way, it is evident that they affect the treatment
## and outcome, meaning that they will be confounders on the future causal
## graph, and that adjusting will be needed.

## It is important to note that for all of the covariates that indicate a 
## relation to needing chemotherapy, most patients with certain characteristics
## still did not need chemotherapy, but these characteristics increase the 
## probability of classifying for chemotherapy. These are not to be exchanged.

