- Project title:

Analyzing the causal effect of chemotherapy on breast cancer recurrence - causal and predictive approaches.


- Project overview:

This project aims to develop a full data science pipeline that enables us to make educated and causal predictions on breast cancer recurrence. The methods and practices developed in the project include, but are not limited to:

1. Directed Acyclic Graph (DAG) construction to model treatment-outcome relationships,
2. Accommodating confounders through different causal methods,
3. Bootstrapped confidence intervals of the Average Treatment Effect (ATE) for hypothesis testing,
4. Consideration of instrumental variables,
5. Causal model refutation,
6. Predictive model usage,
7. Parameter tuning directed cross validation,
8. SHAP values for predictive models.


- Medical problem:

Cancer recurrence is a especially impactful problem within cancer patients, as it directly affects their quality of life and, depending on the situation, chances of survival. Because of this, it is imperative that hospitals and oncology teams correctly conclude on the future of each patient's illness. If this does not hold, the patient and their family will have to incur in high amounts of charges and stress or the risk of death, depending on if the medical professional wrongly concludes for or against recurrence, respectively.

Parting from this problem, this project aims to use causal inference to create educated predictive models and reliably and progressively eliminate this issue.

- Source data description:

The utilized data is a dataset that covers breast cancer patients, and was taken from the cBioPortal for Cancer Genomics.

General information:

1. Even though there are many variables available, this project utilizes the following 15 columns:

'Age at Diagnosis', 'Type of Breast Surgery', 'Cellularity', 'Chemotherapy', 'ER Status', 'HER2 Status', 'Hormone Therapy', 'Inferred Menopausal State', 'Lymph nodes examined positive', 'Neoplasm Histologic Grade', 'PR Status', 'Radio Therapy', 'Relapse Free Status', 'Tumor Size', and 'Tumor Stage',

2. 2509 observations,
3. 5649 missing values,
4. 1346036 bytes of memory usage and,
5. No duplicated values to be concerned about.


- Project structure:


This project consists of 5 stages: data preprocessing, EDA, feature engineering, causal modeling and predictive modeling. Each of these stages consists on importing the used libraries and data, defining the custom functions utilized throughout the file (where applicable),  actual execution of code, and conclusions/exporting necessary files.


- Data cleaning and preprocessing:

See '1_Cleaning' file.

The following steps were taken at this stage:

1. Developed reusable functions for most custom cleaning steps.

2. Wrapped these functions into sklearn-compatible classes enable pipeline integration.

3. Selected the previously mentioned columns.

4. Utilized domain knowledge to impute missing values. The following imputation logic was used: menopausal state based on age, radio therapy based on tumor size and positive lymph nodes, tumor size and positive lymphs based on radio therapy, hormone therapy based on ER and PR status, HER2 based on chemotherapy and neoplasm histologic grade, ER status based on hormone therapy, PR status based on hormone therapy, chemotherapy based on HER2 status, histologic grade based on HER2 status, tumor stage based on tumor size, type of surgery based on tumor size and stage and cellularity based on histologic grade.

Note: Some imputation relied on previously imputed values, traducing possible dependencies or limitations.

5. Adjusted any binary column to be in a yes/no form. Columns with 'Positive' and 'Negative' or 'YES' and 'NO', or any other combination, were all transformed to 'yes' or 'no' values.

6. Reduced dataset memory from 1346036 bytes to 135032 bytes 
This represents a reduction of 1211004 bytes or 89.9 percent. This was done by casting variables with low value cardinalities into 'category' type, while leaving high cardinality variables as 'Object' type.

7. Removed 439 outliers from the variables 'Age at Diagnosis', 'Lymph nodes examined positive' and 'Tumor Size'. This helps models perform better, since the present values are all within a defined and more limited range, and the distributions for these variables' values more closely resemble normal distributions.

8. Exported cleaned data and pipelines as pkl or joblib files.




- Exploratory Data Analysis (EDA):

See '2_EDA' file.

This stage of the project includes some metric analysis, though it is mainly executed visually. It is also important to remark that it is divided in two stages, treatment directed EDA and outcome directed EDA. This was done in order to understand relationships which were relevant to the causal problem and future causal model, instead of the relationships observed being scattered and lacking of actual signal. The purpose of this stage was to help identify potential predictors for the outcome and treatment.

Every section of this stage should be acknowledged as a plot unless 'Metric based.' is written next to the point of interest. Plots can be found on the appropriate folder of the repository.


1. Outcome directed EDA: what seems to be related to recurrence?


a. How many patients did and did not recur? Metric based.

659 patients recurred and 1192 did not.

b. Is tumor aggressiveness (histologic grade) related to recurrence? 

Although higher histologic grades are common amongst recurring patients, they are similarly prevalent in non recurring patients. This points at this variable not being a strong predictor alone for recurrence.

c. Is cancer spread (positive lymph nodes) related to recurrence?

This shows a similar behavior as tumor aggressiveness, trends on recurrence outcome across positive lymph nodes tend to maintain. The only significant difference is that the ratio of low amount of positive lymph nodes is larger for not recurring patients, referencing some signal regarding its correlation with recurrence.

d. Is menopausal state related to recurrence?

In this case, pre menopausal state patient count is approximately equal for either outcome of recurrence, which paired with non-recurrent patients being the majority of the data, proposes that patients with pre menopausal state are more prone to recurring.

The data shows that the proportion of recurring pre menopausal patients is roughly double the proportion of non-recurring pre menopausal patients. This points at pre menopausal state being slightly related to recurrence.

e. Is age related to recurrence?

This plot does not show an age range where recurrence is more likely to happen, only small fluctuations in proportion of recurring vs non-recurring.

f. Is the type of surgery related to recurrence?

No significant relation visibly observed.

g. Is cellularity related to recurrence?

No significant relation visibly observed.

h. Is hormone therapy related to recurrence?

There is a slightly higher proportion of patients that did not undergo hormone therapy amongst recurring patients than amongst non-recurring. The use of hormone therapy can be related to recurrence.

i. Is radio therapy related to recurrence?

No significant relationship visibly observed.

j. Is chemotherapy related to recurrence?

Chemotherapy seems to be related with non-recurrence. This result appears reliable, but it could be because of chemotherapy being effective or because patients that do not need chemotherapy naturally are less likely to recur.


2. Treatment directed EDA: what seems to be related to having chemotherapy?


a. How many patients did and did not undergo chemotherapy? Metric based.

254 did and 1597 did not.

b. Is tumor stage related to chemotherapy?

As opposed to patients that did not undergo chemotherapy, the amount of patients that did surges significantly from stage 2 to stage 3 tumors. In other words, the amount of patients that did not undergo chemotherapy are scattered across tumor stage, but the patients that did need chemotherapy are mostly the ones with stage 3 tumors. This points at tumor stage being significant when defining the need for chemotherapy.

c. Is tumor size related to chemotherapy?

Tumor size is not shown to be an important feature when defining the need for chemotherapy, but sizes of around 24mm undergo chemotherapy more often.

d. Is cancer spread (positive lymph nodes) related to chemotherapy?

Again, even though most patients across all amounts of positive lymph nodes do not undergo chemotherapy, the proportion of patients that do increases rapidly with positive lymph nodes. This also shows to be a strong indicator for needing chemotherapy.

e. Is tumor aggressiveness (histologic grade) related to chemotherapy?

Despite there being more patients with no chemotherapy across all histologic grades, the proportion of patients with chemotherapy increases significantly with this measure.

f. Is cellularity related to chemotherapy?

Cellularity does not seem (visibly) correlated to needing chemotherapy.

h. Is HER2 status related to chemotherapy?

Even though the proportion of patients with HER2 positive status is greater among the patients that underwent chemotherapy, the majority of patients with this treatment still have a negative HER2 status. In different terms, chemotherapy can be applied for every HER2 status, but being HER2 positive increases the probability of having chemotherapy.


3. Conclusions:

It is important to keep in mind that the data is unbalanced both towards the treatment and outcome.

This EDA shows that each available covariate on its own is not enough to predict recurrence, but that these can indicate if a patient will need  chemotherapy - or be assigned to the treatment for causal purposes - which is in turn somewhat related with recurrence.

In this case, even if the covariates do not show a direct effect on the causal outcome, through domain knowledge it is clear that they still are important for it. This way, it is evident that they affect the treatment and outcome, meaning that they will be confounders on the future causal graph, and that adjusting will be needed.

It is important to note that for all of the covariates that indicate a relation to needing chemotherapy, most patients with certain characteristics still did not need chemotherapy, but these characteristics increase the probability of classifying for chemotherapy. These are not to be exchanged.



- Feature engineering:


See '3_Feature Engineering' file.

This file focuses on adapting the cleaned dataset for the models to receive it correctly and be able to complete their tasks, based on the EDA findings. 

This stage consists of the following steps:

1. Encoding the variables according to their specific characteristics.

As covered before, there are some binary variables with 'yes' or 'no' values. Naturally, these were encoded into 1 or 0 depending on their value.

Other columns were categorical and nominal, but did not follow a binary structure, like 'Type of Breast Surgery'. These columns were modified using scikit-learn's One Hot Encoder, leaving them split into several columns. It is important to note that, even though some columns are dropped due to redundancy or to avoid multicollinearity in certain cases, in this situation it was necessary that every column deriving from the variables was included in the data.

The decision to keep every column and assume multicollinearity arose from the nature of these variables. Circling back to surgery type, there were the values of breast conserving or mastectomy. Normally, one of the resulting columns would be dropped to avoid multicollinearity. But in this case, dropping the column for ‘breast conserving surgery’ would make it unclear whether a patient had surgery at all, misleading both the model and the reader. However, in this project, it is made clear that no mastectomy in fact means breast conserving surgery.

This situation makes it necessary to retain all of the deriving columns and accepting the trade off of simplicity and multicollinearity. As a result, this approach allows for greater interpretability and real world application, both crucial for a causal inference model.

Lastly, there are columns with ordinal characteristics. Although the custom encoder developed could be used for ordinal variables, it was not used for this task. The ordinal columns present in the data did not present a monotonic relationship with the causal outcome, and encoding them ordinally would have induced unwanted monotonicity into the models' calculations. These columns were ultimately One Hot Encoded.

2. Second stage memory reduction and functionality improvements:

After encoding every column accordingly, all of the values in the data were numeric, but some columns were still set to category type. The columns that did not need encoding were set to float64.

Because of that, and taking into account the magnitude of the values in the numerical variables, every feature was then set to type float32. This yields an unified data type for models to work with and a slight memory reduction in its second stage.

3. Modifying variable names.

In order for columns to be interpretable by the posteriorly implemented models, their names needed to not contain any spaces. A simple for loop was used to replace these by '_'.



After these processes the data is now ready for modeling.




- Causal section:

See '4_Causal' file.

Definitions:

Confounders: Variables that affect both the treatment and outcome. 
Colliders: Features affected by 2 or more other features.
Mediators: Variables that are only affected by the treatment and affect the outcome.
Instrumental Variables (IV): The ones that affect the outcome only through its effect on the treatment.

Causal question: How does chemotherapy affect breast cancer recurrence?

This file consists of several steps are that crucial for causal inference. See below.

1. Simplifying Data:

In order to make this process smoother and simpler, some unnecessary columns are dropped and some column names are simplified.

2. Causal DAG Creation:

The directed acyclic graph (DAG) is created from a string representation based on domain knowledge. This DAG was constructed using domain knowledge about breast cancer recurrence mechanisms.

3. Creating Causal Model:

The causal model is created with DoWhy's Causal Model object, using a copy of the original data, chemotherapy as the treatment, recurrence as the outcome and the previously defined string as the graph.

4. Causal Graph Visualization and Interpretation (See plot):

Since DoWhy returns a very poor representation of this specific graph, networkx was used to make a more appropriate visual representation of it. Using the graph, conclusions about confounders, colliders and mediators are drawn. These were confirmed by DoWhy and will be presented and explained below.

5. Estimand Identification:

Used DoWhy to identify estimands. The analysis returned the following.

Confounders needing back-door adjustment: menopausal state (now divided into pre and post by columns), histologic grade, positive lymph nodes and tumor size and stage.

Potential Instrumental Variables: surgery type (now divided into two columns as well), hormone therapy and HER2, ER and PR statuses. 

No colliders, mediators or the need for front door adjustments were identified.


6. Effect Estimation - Linear Regression:

The average treatment effect (ATE) through the linear regression method was of ~0.1. This means that chemotherapy has a limited yet potentially counterproductive causal effect on patients.

This does not align with the medical knowledge and practices. A possible reason for this, which was briefly covered during the EDA, is that patients with chemotherapy naturally have more delicate conditions, which is why they tend to recur, but also to be a subject to chemotherapy. This would result in patients with chemotherapy recurring more often than patients that did not make use of the treatment, but not because this treatment has this type of causal effect, but due to other features. In other words, this ATE is likely biased and does not serve as a causal conclusion.

Since this effect was estimated after adjusting for the present confounders, it points to other hidden confounders. The ignorability assumption was tested, and the results will be explained in further stages of the project.

Nevertheless, a normal measure to take after this finding is covered in the next step.

7. Simpson's Paradox Check (See plot):

Due to our ATE not aligning with medical knowledge, Simpson's paradox is suspected. This check divides the data by the amount of positively tested lymph nodes, since this is a measure for cancer spread, which is an important measure for recurrence.

The plot shows the presence of this paradox by revealing that the correlation of lymph nodes and recurrence is heterogenic. Patients with 0 positive lymph nodes appear to benefit the most from chemotherapy. In order to test this, the next step follows.

8. Effect on 0 positive lymph nodes.

Using backdoor linear regression again, the conditional average treatment effect (CATE) for patients with 0 positively tested lymph nodes is approximately 0.31, akin to the ATE for the whole population. From this, two main conclusions can be drawn. 

First, even if the Simpson's paradox is present, it does not go further than visual correlation, since the estimated causal effect of chemotherapy for this section of the population is still counterintuitive. Second, after analyzing previous steps, the first conclusion reinforces the suspicion that there likely is a hidden confounder affecting the experiment's results.

9. Estimating effect through other methods.

The ATE, adjusting for the same confounders was estimated through Propensity Score Matching (PSM) and Inverse Probability Weighting (IPW), and the results were of roughly 0.21 an 0.13 respectively. 

These values are counterintuitive as well, pointing at linear regression not lacking robustness or insight, but a deeper problem. Again, likely the violation of the ignorability assumption.

For further tests and measures the ATE from IPW will be used, since it is the median of the three measures.

10. Testing the ATE's statistical significance - Bootstrapped confidence intervals and hypothesis testing (See distribution plots).

This step consists of conducting a hypothesis test at a 5% level of significance, and using confidence intervals to test whether the resulting ATE's are statistically significant. One hundred iterations were performed for every method.

Hypothesis test setup:

Null hypothesis: ATE = 0
Alternative hypothesis: ATE > 0 (since we intend to test the specific result acquired)
Significance level: 5%

95% Confidence Intervals for ATE's:

Linear Regression: (0.03, 0.2)
PSM: (-0.06, 0.29)
IPW: (0.04, 0.2)

Conclusions:

Since the CI for PSM contains 0, we fail to reject the null hypothesis at the 5% level of significance, meaning that there is not enough statistical evidence to say that the true ATE is greater than 0 for this method, and making it the least statistically significant estimation. 

For the other two methods, since 0 is not contained in the intervals, the null hypothesis is rejected at the 5% level of significance, and there is statistically significant evidence that the ATE's are greater than 0.

Given that the result from IPW is our general measure, we assume statistical significance on the resulting ATE.


11. Considering Instrumental Variables:

Using the causal model, EDA and domain knowledge, the candidates to act as IV's are PR, ER or HER2 statuses. Still, in order to check if they have a statistically significant relationship with the treatment, tests need to be made.

In this project linear regression models from statsmodels are used to assess the significance of the relationships between each of the variables and the treatment. In the following lines an F statistic with a low p-value is considered as statistically significant.

a. PR Status: the model returns a R2 value of 0.09 with a p-value of the F statistic smaller than 0.01. This means that these variables are very weakly related and that this is statistically significant.

b. ER Status: this model shows a statistically significant R2 of 0.21. This is an improvement from the last one, but still not strong enough to justify strong association.

c. HER2 Status: Again, the model reveals an R2 of 0.035 with high significance.

From these experiments it is clear that none of these variables are fit instruments, and that the local average treatment effect (LATE) should not be computed through any of them.


12. Sensitivity analysis - refutation:

The purpose of this section is to assess whether the findings and unexpected ATE are significant and reliable given this specific model (not a real scenario). 

In order to evaluate this, the model will be tested by: adding an unobserved random hidden confounder, which will indicate if the model is sensitive to them; using a placebo treatment, to see if the model has been detecting purely spurious or actual causal relationships; and using a subset of the data, in order to understand if the model based its conclusions parting only from a subset of the data, and not the whole group.

These refutations are done using DoWhy's 'estimate_refuter' with different methods.

a. Random confounder:

After introducing a random confounder and re estimating the ATE, a variation of around 100% is presented. This means that the originally produced ATE is not robust or reliable by any means. Additionally, given the nature of the refutation method utilized, it points at the model being sensitive to unobserved confounders, meaning that there probably is one, and that the ignorability assumption was indeed violated.

b. Placebo treatment:

With a random placebo treatment introduced, the model was not able to estimate any effect on the outcome. This suggests that the treatment truly has a causal, rather than spurious or observational, impact on the outcome and that the issue relies in another point (point number 1).

c. Subset priority:

This test returned an ATE very similar to the originally observed, but with a very high p-value (0.94). Similar ATE's typically mean that the model is taking into account all of the data to make conclusions, but the high p-value reveals that this could have happened by chance, and not by the model's or situation's merit. Overall, this is test is not statistically significant and no conclusions or assumptions coming from it should be taken as true.


These analyses show that the misleading ATE's found before could be due to the model basing its conclusions over a partition of the data, but most importantly and likely, on the absence of an important confounder not found in the data.

It is also important to know that, while not tested, the imbalances of the treatment and outcome on the raw data point to the violation of the assumption related to coverage.


13. Conclusions:

From this section it can be noted that the estimated effect of chemotherapy is not consistent with medical practices or with the observational data (see section on Simpson's paradox), as it points at chemotherapy causally leading to cancer recurrence. This was also shown to be statistically  significant within this model, meaning that there are deeper issues than model execution or random states. Additionally, the use of IV's was not viable in this case since the medically possible variables showed a poor statistical relationship with the treatment, and would not be reliable when estimating its causal effect on the outcome.

However, the implemented practices show that this is due to the violation of the ignorability assumption, which states that there must be no unobserved confounders, which is due to a limitation of the data. These inconsistencies could have been discovered through balance checks, but the underlying causes already seem evident.

In further experimentations, it would be beneficial to infer causal effects through a richer dataset or randomized control trials (RCT).



- Predictive section:


See '5_Predictive' file.

This section focuses on the usage of predictive models in order to determine whether a patient will recur or not. The goal was to develop effective predictive models while acknowledging the priorities imposed by a medically sensitive setting.

Originally, the findings from the causal section of this project were intended to serve as guidance point to build the predictive models. However, due to the causal estimates contradicting expected treatment behavior, these predictive models are developed independently of the prior causal analysis.

Model considered: decision tree, random forest, XGBoost, logistic regression.

These models were chosen due to their nature as classifiers. The first 3 serve as an effectiveness comparison of similar but distinct methods, and the fourth acts as a different approach to the problem, parting from logistic regression being convenient as a binary classifier.

It is important to consider that, in this situation, a model incurring in false negatives is the worst possible outcome since that could often result in the death of the patient. Because of this, recall will be the main metric used when addressing the models' performance.



The following processes were carried out at this stage:


1. Splitting the data into train and test data frames:

Using scikit-learn's function, the data was partitioned into 70% for training the models, and the other 30% for testing their performance. This leaves 1295 records for training and 556 for testing. 

This results in two feature matrices and two corresponding outcome vectors for training and testing.

2. Parameter selection through cross validation:


In order to find the optimal parameter configuration for every model, cross validation was carried out. 

The following are a few considerations for this step:

a. The models have up to seven hyperparameters to be considered, each with around 4 possible values. In order to reduce computational costs, but to keep the benefits from a  cross validated parameter selection, a randomized grid search will be applied instead of an exhaustive one. 

This search will yield results that are very close to optimal (or the optimal values in some cases), with a significantly reduced computational effort.

b. As stated before, the dataset presents class imbalance in the outcome variable, with recurrence cases being underrepresented. At this stage, that is significant enough to select the test data of the cross validations using stratified folds.

This will ensure that, in no occasion, the data's imbalance will play a role to favor or harm a specific set of parameters' performance.

c. In order to further account for this imbalance, the models were created to consider it through the 'class_weight' or 'scale_pos_weight' parameter as needed.

d. As extra details, 5 data folds and 20 iterations of the grid search were performed.

e. The following are the parameter grids for every model:

Decision Tree:

dt_param_grid = {
    'max_depth':[5, 10, 15, None],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf':[1, 2, 4, 8],
    'max_features':['sqrt', 'log2', None]}


Random Forest:

rf_param_grid = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "max_samples": [0.5, 0.7, 0.9, None]}


XGBoost:

xgb_param_grid = {
    "n_estimators": [50, 100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "max_depth": [5, 10, 15, None],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.5, 0.7, 0.9, 1],
    "gamma": [0, 0.1, 0.3, 1],
    "scale_pos_weight": [1, 3, 5, 10]}


Logistic Regression:

lr_param_grid = {
    'C':[0.01, 0.1, 1, 10, 100],
    'penalty':['l1', 'l2']}




After completion of the process for every model, the best parameters and recall score were printed in the python script and added as in-line comments, and the best performing model was initialized to further test on the 30% of data destined for this purpose.



3. Testing models:

Each model was tested on recall (on the recurring class, in order to focus on the potentially most affected patients), accuracy, their confusion matrix and ROC-AUC score. Each of these metrics serves to account for the false negatives, understand how often the model predicts correctly, have raw data on the models performance, and diagnose the models' ability to intentionally separate outcome categories.

The following are the results for each model:


Decision Tree:

Recall on recurrence class: 0.54
Accuracy: 0.55
Confusion Matrix: 200 TP, 158 FP, 92 FN, 106 TN
ROC-AUC: 0.54


Random Forest:

Recall on recurrence class: 0.43
Accuracy: 0.59
Confusion Matrix: 240 TP, 118 FP, 112 FN, 86 TN
ROC-AUC: 0.55


XGBoost:

Recall on recurrence class: 0.95
Accuracy: 0.39
Confusion Matrix: 31 TP, 327 FP, 10 FN, 188 TN
ROC-AUC: 0.51


Logistic Regression:

Recall on recurrence class: 0.57
Accuracy: 0.56
Confusion Matrix: 199 TP, 159 FP, 86 FN, 112 TN
ROC-AUC: 0.56



Conclusions:

These evaluations show that the simplified recall is misleading in the apparently best model. The XGBoost model has a recall of 95% on the recurrence class, but the confusion matrix reveals that it is simply predicting recurrence in the vast majority of cases. This makes the model good at predicting when a patient will be recurring but condemns healthy patients to more severe treatments, making it unreliable in a real medical setting.

From all of the metrics, the model that best balances the important statistics is the logistic regression model. It unites the best trade-off between recall and accuracy, as well as the highest ROC-AUC score from the 4 models.

Even though none of the models performed as expected, the logistic regression model is chosen as the best option.


A summary of the important metrics for each model is available as 'summary' on this python script.


4. Re training chosen model:


As explained before, the data was split into training and testing sets for practicality purposes and to avoid possible data leakage.

Now, after identifying the best model (Logistic Regression) and its best configuration, the model was retrained using 100% of the data in order to maximize its exposure to the available information.

The goal of this stage is for the model to learn from all of the available records and resources, making it more fit when developing the same task under newly unseen data. Otherwise, there would have been model performance to be improved, and this measure ensures that is translated from simply potential into actual performance.


5. Model interpretation and explanation - SHAP values (See plots):


To investigate the factors influencing the model's decisions, SHAP (SHapley Additive exPlanations) values were used and interpreted through plots. This method offers a much more interpretable perspective of the models functioning, enriching our knowledge regarding its predictive algorithm and possible future interest points.

The SHAP values were used to produce a summary plot and a waterfall plot. These plots serve to understand how the model generally decides on every observation, and to visualize how classification was concluded for a single specified observation respectively.

The summary plot shows Age at Diagnosis, ER Status and positive lymph nodes to be the defining features for classification (class not specified).

On the other hand, the waterfall plot for observation number 1 reveals that the age at diagnosis is what pushes the model the most to conclude on recurrence. Opposite to general expectations, tumor size has an inverse relationship with recurrence for this specific prediction.

These plots served to understand the models decision making beyond simple numerical metrics, allowing clinicians to understand why models make certain predictions. This, and the medical professional's conclusions on the matter, can aid data scientist to produce more reliable models, as flaws are easily highlighted and significantly more approachable.




- Conclusions and further recommendations:

This project aimed to develop a full data science pipeline, from data collection and cleaning to causal and predictive modeling, where causal conclusions would be implemented into predictive models for added reliability and interpretability. After completing the causal section, it became clear that the estimated treatment effects didn’t fully reflect the underlying medical theory, so the predictive models had to be developed independently.

Still, these models were implemented, evaluated, and interpreted thoroughly to ensure the most accurate and meaningful results.

Although scikit-learn pipelines were originally built to promote modularity, reproducibility, and ease of deployment, they were ultimately set aside during this version of the project due to integration challenges. However, the pipeline structures remain part of the codebase and can serve as a foundation for future refactoring or deployment-oriented iterations where pipeline integration becomes more practical.

Finally, it’s clear that the project would have benefited from a richer dataset, more domain-specific knowledge, or ideally, data from a randomized control trial (RCT). Any of these would have supported more informed decisions and helped lead to a statistically significant ATE that aligned more closely with medical expectations.
