# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:50:05 2025

@author: nicol
"""

import pandas as pd
import numpy as np
import joblib
from dowhy import CausalModel
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

np.random.seed(42)

clinic_data = pd.read_pickle("C:/Users/nicol/OneDrive/Desktop/Predicting cancer recurrence through causal inference/clinic_data_engineered.pkl")
    

        ## Used functions:
            
def estimate_effect_fun(model, estimand, method):
    
    estimate = model.estimate_effect(estimand, method_name=method,
                                     method_params={"random_state": 42})
    
    return estimate

def bootstrapping(iters, data, method):
    
    bootstrap_ates = []
    
    for i in range(iters):
        print(f"Bootstrap iteration {i+1}/{iters}")
        
        sample = data.sample(n=len(data), replace=True).copy()
        model = CausalModel(sample, 'Chemotherapy', 'Recurrence', graph=graph)
        estimand = model.identify_effect()
        estimate = estimate_effect_fun(model, estimand, 
                                             method)
        bootstrap_ates.append(estimate.value)
        
        
    return bootstrap_ates


def estimate_refuter(model, estimand, estimate, method):
    
    refuted = model.refute_estimate(estimand, estimate, method_name=method)
    
    return refuted
    

        ## Plot settings
    
sns.set_style('darkgrid')
sns.set_context('talk')
sns.set_palette('deep')



## Taking into account the previous EDA and domain knowledge, the Cellularity
## and Radio Therapy covariates need to be removed.

clinic_data.drop(columns=['Cellularity_High', 'Cellularity_Moderate',
                          'Cellularity_Low', 'Radio_Therapy'], inplace=True)

## In this new list of variables there are 2 columns for the type of surgery 
## menopausal state. These are needed, since the columns cannot be collapsed
## into asingle column as they are categorical (and the models need numeric
## variables), and they don't keep monotonicity towards the outcome, meaning
## assigning numeric values can be misleading.

## Still, to visualize the DAG, their names will be simplified.

clinic_data.rename(columns={
    'Type_of_Breast_Surgery_BREAST_CONSERVING':'Breast_Conserving',
    'Type_of_Breast_Surgery_MASTECTOMY':'Mastectomy',
    'Inferred_Menopausal_State_Post':'Meno_Post',
    'Inferred_Menopausal_State_Pre':'Meno_Pre'}, inplace=True)

## With this, define the causal graph:
    
graph = """

digraph {
    Age_at_Diagnosis;
    Chemotherapy;
    ER_Status;
    HER2_Status;
    Hormone_Therapy;
    Lymph_nodes_examined_positive;
    Neoplasm_Histologic_Grade;
    PR_Status;
    Recurrence;
    Tumor_Size;
    Tumor_Stage;
    Breast_Conserving;
    Mastectomy;
    Meno_Post;
    Meno_Pre;
    
    
    Breast_Conserving -> Chemotherapy
    Mastectomy -> Chemotherapy
    ER_Status -> Chemotherapy
    HER2_Status -> Chemotherapy
    PR_Status -> Chemotherapy
    Hormone_Therapy -> Chemotherapy
    Meno_Post -> Chemotherapy
    Meno_Pre -> Chemotherapy
    Lymph_nodes_examined_positive -> Chemotherapy
    Neoplasm_Histologic_Grade -> Chemotherapy
    Tumor_Size -> Chemotherapy
    Tumor_Stage -> Chemotherapy
    
    Age_at_Diagnosis -> Recurrence
    Chemotherapy -> Recurrence
    Meno_Post -> Recurrence
    Meno_Pre -> Recurrence
    Lymph_nodes_examined_positive -> Recurrence
    Neoplasm_Histologic_Grade -> Recurrence
    Tumor_Size -> Recurrence
    Tumor_Stage -> Recurrence
    
    
    }
"""


causal_model = CausalModel(clinic_data.copy(), ["Chemotherapy"],
                           ["Recurrence"], graph=graph)


## Graph visualization (the DoWhy tool returns a very poor visualization):

G = nx.DiGraph()

edges = [
    ("Breast_Conserving", "Chemotherapy"),
    ("Mastectomy", "Chemotherapy"),
    ("ER_Status", "Chemotherapy"),
    ("HER2_Status", "Chemotherapy"),
    ("PR_Status", "Chemotherapy"),
    ("Hormone_Therapy", "Chemotherapy"),
    ("Meno_Post", "Chemotherapy"),
    ("Meno_Pre", "Chemotherapy"),
    ("Lymph_nodes_examined_positive", "Chemotherapy"),
    ("Neoplasm_Histologic_Grade", "Chemotherapy"),
    ("Tumor_Size", "Chemotherapy"),
    ("Tumor_Stage", "Chemotherapy"),
    
    ("Age_at_Diagnosis", "Recurrence"),
    ("Chemotherapy", "Recurrence"),
    ("Meno_Post", "Recurrence"),
    ("Meno_Pre", "Recurrence"),
    ("Lymph_nodes_examined_positive", "Recurrence"),
    ("Neoplasm_Histologic_Grade", "Recurrence"),
    ("Tumor_Size", "Recurrence"),
    ("Tumor_Stage", "Recurrence"),
]

G.add_edges_from(edges)

node_colors = []
for node in G.nodes:
    if node == "Chemotherapy":
        node_colors.append('lightgreen')
    elif node == "Recurrence":
        node_colors.append('lightcoral')
    else:
        node_colors.append('skyblue')

plt.figure(figsize=(14, 10))
layout = nx.spring_layout(G, seed=22)

nx.draw(
    G,
    layout,
    with_labels=True,
    node_size=3000,
    node_color=node_colors,
    font_size=10,
    font_weight='bold',
    arrows=True,
    arrowsize=20
)

plt.title("Causal Graph", fontsize=16)
plt.show()




## With this graph, it is revealed that the following covariates are confounders:
## Tumor size, tumor stage, histologic grade, positive lymph nodes and 
## pre and post menopusal state. No mediators or colliders are shown, but there
## are some covariates that could serve as instrumental variables.


## In order to see which variables DoWhy recommends adjusting for:
    
identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)

print("Identified estimand:", identified_estimand)


## The identified estimand shows that menopausal states, histologic grade,
## positive lymph nodes and tumor size and stage are confounders that need
## adjusting, and that type of surgery, hormone therapy and HER2, ER and PR
## statuses can be used as instrumental variables in order to estimate
## the causal effect of chemotherapy on recurrence. It is also described that
## there are no frontdoor paths to be addressed.



## Estimating effects:
            
## Base measure, ATE on treatment adjusting for confounders.

linear_estimate = estimate_effect_fun(causal_model,
                                      identified_estimand, 
                                      'backdoor.linear_regression')




print('ATE, linear regression:',linear_estimate.value) ## ~0.1



## This points at 9.9% more probability of being a recurring patient if
## chemotherapy is taken than if not, which does not make medical sense.
## Still, this could be due to more severe types of cancer needing more intense
## treatments like chemotherapy. In this case the Simpson's paradox must be 
## checked to find validate this result.

## Positive lymph nodes

clinic_data['Lymph_nodes_bin'] = pd.cut(
    clinic_data['Lymph_nodes_examined_positive'], bins=[-1, 0, 3, 10, 50], 
    labels=['0', '1-3', '4-10', '11+']
)

sns.barplot(data=clinic_data, x='Lymph_nodes_bin', y='Recurrence', 
            hue='Chemotherapy', estimator='sum')
plt.title('Recurrence by Chemotherapy across Lymph Node Positivity Levels')
plt.ylabel('Recurrence Count')
plt.xlabel('Lymph Node Positive Group')
plt.show()

## This plot shows that the effectiveness of chemotherapy is heterogenic
## across the amount of positive lymph nodes. More specifically, patients
## with 0 positive nodes benefit a lot more from chemotherapy than the ones
## with more lymph nodes

## Testing CATE at 0 positive lymph nodes.

no_nodes = clinic_data.loc[clinic_data['Lymph_nodes_examined_positive'] == 0,
                           :]

no_nodes_model = CausalModel(no_nodes, ["Chemotherapy"],
                           ["Recurrence"], graph=graph)

no_nodes_estimand = no_nodes_model.identify_effect()

no_nodes_estimate = estimate_effect_fun(no_nodes_model,
                                        no_nodes_estimand, 
                                        'backdoor.linear_regression')

print('CATE for no positive lymph nodes:', no_nodes_estimate.value) ## ~0.31

## This graph and causal test shows that even though the Simpson's paradox is 
## present in the data, the causal effect for the subsets is still positive,
## akin to the treatment's effect on the whole population.

## In any case, the following code and its result suggest selection bias:
    
clinic_data.groupby('Chemotherapy')['Recurrence'].mean() ## A higher average 
## of recurring patients that use chemotherapy against the patients that
## recurred without chemotherapy.


## Other backdoor methods:
    
psm_estimate = estimate_effect_fun(causal_model,
                                      identified_estimand, 
                                      'backdoor.propensity_score_matching')



ipw_estimate = estimate_effect_fun(causal_model,
                                      identified_estimand, 
                                      'backdoor.propensity_score_weighting')


print('ATE, propensity score matching:', psm_estimate.value) ## ~0.21
print('ATE, inverse probability weigthing (or propensity score weigthing):',
      ipw_estimate.value) ## ~0.13


## These results are contradicting, as they express that chemotherapy leads
## to recurrence. Even after testing for the Simpson's paradox the causal
## effect does not match the visual observations. Other causal estimators will
## be considered.

## For further methods and considerations the ATE from IPW will be used since
## it is the closest to the average ATE from the 3 methods.


## Bootstrapping: 95% CI for ATE and histogram.

linear_boots = bootstrapping(100, clinic_data, 'backdoor.linear_regression')
print('95% CI linear:', np.percentile(linear_boots, [2.5, 97.5])) ## 0.03 - 0.2

sns.histplot(x=linear_boots, kde=True)
plt.title('Linear_boots distribution')
plt.xlabel('ATE')
plt.tight_layout()
plt.show()

psm_boots = bootstrapping(100, clinic_data, 'backdoor.propensity_score_matching')
print('95% CI psm:', np.percentile(psm_boots, [2.5, 97.5])) ## -0.06 - 0.29

sns.histplot(x=psm_boots, kde=True)
plt.title('PSM_boots distribution')
plt.xlabel('ATE')
plt.tight_layout()
plt.show()

ipw_boots = bootstrapping(100, clinic_data, 'backdoor.propensity_score_weighting')
print('95% CI IPW:', np.percentile(ipw_boots, [2.5, 97.5])) ## 0.04 - 0.2

sns.histplot(x=ipw_boots, kde=True)
plt.title('IPW_boots distribution')
plt.xlabel('ATE')
plt.tight_layout()
plt.show()


## Through a hypothesis test with null: ATE = 0 and alternative: ATE > 0 and at
## the 5% level of significance, the null hypothesis can only fail to be rejected
## for the PSM method, and by a small margin. For the other methods, the null
## hypothesis needs to be rejected, meaning that there is enough statistical
## evidence to say that the true ATE differs from 0. Additionally, it can be
## seen that it is actually higher than 0 in this case.

## Summary oh hyp test:
## - Linear regression and IPW show statistically significant ATE's
## - PSM includes 0 in the positive interval which leads to rejecting the null



## IV's

## The appropriate medical and statistical instrumental variables are PR Status,
## ER Status and HER2 Status. These need to be tested for significance against
## the treatment.

pr_model = smf.ols('Chemotherapy ~ PR_Status', data = clinic_data).fit()
print(pr_model.summary()) 
## pr_model shows an R2 of 0.09 and a p-value < 0.01, meaning that these are
## not correlated, and this relationship is statistically significant.

er_model = smf.ols('Chemotherapy ~ ER_Status', data = clinic_data).fit()
print(er_model.summary())
## The R2 for this model is of 0.21, showing a stronger relationship, but still
## not strong enough to estimate causal effect, and with high significance too.

her2_model = smf.ols('Chemotherapy ~ HER2_Status', data = clinic_data).fit()
print(her2_model.summary())
## The relationship is weak, with R2 of 0.035 and highly significant.

## According to these models none of these variables are reliable or statistically
## significant instrumental variables and should not be used as such.



## Sensitivity analysis

## In order to assess the results for robustness and bias, the model will be
## tested on ignorability, propensity to finding only spurious relationships
## and stability/the model basing causality on a subset of the data.

## Ignorability:
    
refute_ignor = causal_model.refute_estimate(
    identified_estimand, ipw_estimate, method_name='add_unobserved_common_cause')

refute_ignor = estimate_refuter(causal_model
                                , identified_estimand
                                , ipw_estimate
                                , 'add_unobserved_common_cause')

print(refute_ignor) ## An ATE variation of around 100% (depending on the seed),
## meaning that the original ATE is not robust at all and probably biased, and
## there is likely a hidden confounder not included on the data but that affects
## the outcome.

## Spurious (non causal) relationships:
    
refute_placebo = estimate_refuter(causal_model
                                , identified_estimand
                                , ipw_estimate
                                , 'placebo_treatment_refuter')

print(refute_placebo) ## The refuter was not able to find an effect using a 
## placebo, meaning that the model is not basing its conclusions on spurious
## relationships.


## Subset priority:

refute_subset = estimate_refuter(causal_model
                                , identified_estimand
                                , ipw_estimate
                                , 'data_subset_refuter')

print(refute_subset) ## This shows a similar effect to when the model uses
## all of the data in order to test causality, but the p-value of 0.94
## shows that this is not statistically significant, meaning that the
## model likely is basing its decisions on a specific subset of the data.


## These analyses show that the misleading ATE's found before could 
## be due to the model basing its conclusions over a partition of the data,
## but most importantly and likely, on the absence of an important confounder
## not found in the data.




## From this file it can be noted that the estimated effect of chemotherapy
## is not consistent with medical practices or with the observational data
## (see section on Simpson's paradox), as it points at chemotherapy causally
## leading to cancer recurrence. This was also shown to be statistically 
## significant within this model, meaning that there are deeper issues than
## model execution or random states. Additionally, the use of IV's was not
## viable in this case since the medically possible variables showed a poor
## statistical relationship with the treatment, and would not be reliable when
## estimating its causal effect on the outcome.

## However, the implemented practices further in the file show that this
## is due to the violation of the ignorability assumption, which states
## that there must be no unobserved confounders, which is due to a limitation
## of the data. These inconsistencies could have been discovered through
## balance checks, but the underlying causes already seem evident.

## In further experimentations, it would be beneficial to infer causal
## effects through a richer dataset or randomized control trials (RCT).


## Reusable causal model as object:
    
joblib.dump(causal_model, 'causal_model.joblib')

