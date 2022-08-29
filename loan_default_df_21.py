#!/usr/bin/env python
# coding: utf-8

# # Hacker Earth Machine Learning Challenge 
# 
# ## Problem Statement
# The Bank Indessa has not done well in last 3 quarters. Their NPAs (Non Performing Assets) have reached all time high. It is starting to lose confidence of its investors. As a result, it’s stock has fallen by 20% in the previous quarter alone.
# After careful analysis, it was found that the majority of NPA was contributed by loan defaulters. With the messy data collected over all the years, this bank has decided to use machine learning to figure out a way to find these defaulters and devise a plan to reduce them.
# This bank uses a pool of investors to sanction their loans. For example: If any customer has applied for a loan of $20000, along with bank, the investors perform a due diligence on the requested loan application. Keep this in mind while understanding data.
# In this challenge, you will help this bank by predicting the probability that a member will default.
# 
# 
# ### Solution Approach
# 

# # Importing needful libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import catboost as cat_
import seaborn as sns
import lightgbm as lgb
from sklearn.impute import SimpleImputer as Imputer
from sklearn import preprocessing
import re
import timeit
import random
random.seed(3)


# In[2]:


dataset = pd.read_csv("datasets/dataset.csv")


# In[3]:


#dataset = dataset.sample(frac = 0.1, random_state=10)


# In[4]:


#pd.DataFrame(dataset).to_csv('datasets/dataset.csv', index=False)


# In[5]:


dataset.shape


# In[6]:


#member_id = test.member_id


# ## Merging the train and test data
ntrain = train.shape[0] 
ntest = test.shape[0]
dataset = pd.concat((train, test), sort=False).reset_index(drop=True)
# In[7]:


y = dataset.loan_status.values
y.shape


# ## VISUALISATION

# In[8]:


plt.figure(figsize = [12,8])
sns.countplot(x = 'loan_status', data = dataset)


# In[9]:


plt.figure(figsize=(22,8))
sns.countplot(x = 'purpose', data = dataset)


# In[10]:


#dataset['home_ownership'].values_count()


# In[11]:


plt.figure(figsize=(20,8))
sns.countplot(x = 'home_ownership', data = dataset)


# In[12]:


plt.figure(figsize=(20,10))
#cor = dataset.
cor = dataset.drop('member_id', axis=1).corr(method = 'spearman')
fig = sns.heatmap(cor, annot=True)
plt.show(fig)


# In[13]:


#correlation, pval = spearmanr(dataset.drop("member_id",axis=1))
#print(f'correlation={correlation:.6f},p-value={pval:.6f}')
from scipy.stats import rankdata
sns.jointplot(x=dataset['emp_length'], y=dataset['loan_amnt'])


# In[14]:


print(dataset.drop("member_id", axis=1).corr(method='spearman'))


# ## DATA CLEANING 

# ## Droping some unwanted dependent variables

# In[15]:


dataset = dataset.drop(['member_id','total_rec_int','batch_enrolled','funded_amnt_inv','title','grade', 'sub_grade','desc', 'zip_code','addr_state','emp_title','verification_status', 'pymnt_plan','delinq_2yrs',
       'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record','open_acc', 'pub_rec','initial_list_status','total_rec_late_fee',
       'recoveries', 'collection_recovery_fee','tot_coll_amt', 'collections_12_mths_ex_med', 'mths_since_last_major_derog','verification_status_joint', 'last_week_pay','acc_now_delinq','loan_status'], axis=1)


# In[16]:


dataset.shape


# In[17]:


dataset.emp_length.unique()


# In[18]:


dataset['emp_length'] = dataset['emp_length'].replace({np.nan:0, '< 1 year':1, '1 year':2, '2 years':3, '3 years':4, '4 years':5, '5 years':6, '6 years':7, '7 years':8, '8 years':9, '9 years':10, '10+ years':11})


# In[19]:


dataset.emp_length.unique()


# In[20]:


dataset.head()


# In[21]:


dataset.columns


# In[22]:


dataset.application_type.unique()


# In[23]:


for key, value in dataset.items():
    dataset['term'] = dataset['term'].replace({'36 months':36, '60 months':60})
    dataset['home_ownership']=dataset['home_ownership'].replace({'OWN':1, 'MORTGAGE':2, 'RENT':3, 'OTHER':4, 'NONE':5, 'ANY':6})
    dataset['purpose'] = dataset['purpose'].replace({'small_business':1,'debt_consolidation':2, 'home_improvement':3, 'credit_card':4,'major_purchase':5,  
                                                     'vacation':6, 'car':7, 'moving':8,'medical':9, 'wedding':10, 'renewable_energy':11, 'house':12, 'educational':13,'other':14,})
    dataset['application_type'] = dataset['application_type'].replace({'INDIVIDUAL':1, 'JOINT':2})
    


# ## Filling missing values

# In[24]:


dataset.isnull().sum()


# ## New features

# In[25]:


value = -9999
def helping_features(value):
    i = ((dataset['int_rate'])/100)/12
    d = 1 - (1/(1+i)**dataset['term'])
    dataset['Monthly_supposed_payment'] = (dataset['funded_amnt']*i)/d
    dataset['Total_refund'] = dataset['Monthly_supposed_payment']*dataset['term']
    dataset['Interest_amnt'] = dataset['Total_refund'] - dataset['funded_amnt']
    dataset['Monthly_income'] = dataset['annual_inc'].apply(lambda x : x/12 if x >=0 else -9999)


# In[26]:


#helping_features(-9999)
#dataset.head()
#dataset['unpaid_%'] = (dataset['Interest_amnt'] - dataset['total_rec_int'])/dataset['Interest_amnt']

#dataset['last_to_term'] = dataset['last_week_pay']/(dataset['term']*52/12)

#dataset['monthly_int_to_debt_repay'] = dataset['Interest']/dataset['Monthly_debt_repay']/12

#dataset['iti'] = dataset['Interest_amnt']/dataset['Monthly_income']/12

#dataset['dti_and_iti'] = dataset['iti'] + dataset['dti']

#dataset['dtc'] = dataset['Monthly_debt_repay']/dataset['total_rev_hi_lim']/12

#dataset['itc'] = dataset['Interest_amnt']/dataset['total_rev_hi_lim']/(12*12)


# In[27]:


helping_features(-9999)
dataset.head()


# In[28]:


le = preprocessing.LabelEncoder()
for col in dataset.columns:
    dataset[col] = le.fit_transform(dataset[col])


# In[29]:


def fill_nulls(value):
    cols_fill = ['annual_inc','revol_util', 'total_acc', 'tot_cur_bal', 'total_rev_hi_lim']
    
    if value == -9999:
        for col in cols_fill:
            dataset.loc[dataset[col].isnull(), col] = -9999
    else : 
        for col in cols_fill:
            dataset.loc[dataset[col].isnull(), col] = dataset[col].median()


# In[30]:


fill_nulls(-9999)


# In[31]:


dataset.columns


# In[32]:


dataset.info()


# In[33]:


np.any(np.isnan(dataset))
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)


# ## SCALING THE DATASET 

# In[34]:


# Use 3 features
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif
df = SelectKBest(f_classif, k=3)
df1 = StandardScaler()
data_scale = df1.fit_transform(dataset) 
data = data_scale


# In[35]:


columns = dataset.columns


# In[36]:


data_df = pd.DataFrame(data, columns=columns)
data = data_df
data.head()


# ### Spliting the test and trian data

# In[37]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,y, test_size=0.2, random_state=0)


# In[38]:


X_train.shape


# In[39]:


X_test.shape


# In[40]:


y_train.shape


# In[41]:


y_test.shape


# # Training the Data - Model

# In[42]:


# lightgbm for classifier
start = timeit.default_timer()
from lightgbm import LGBMClassifier
from matplotlib import pyplot

# evaluate the model
#model = LGBMClassifier()
# fit the model on the whole dataset
model_lgb = LGBMClassifier()
model_lgb.fit(X_train, y_train)
y_pred_lgb = model_lgb.predict(X_test)
y_pred_prob_lgb = model_lgb.predict_proba(X_test.values)[:,1]
stop = timeit.default_timer()
print(stop - start)


# In[43]:


y_pred_lgb


# In[44]:


y_pred_prob_lgb


# In[45]:


# catboost for classification
start = timeit.default_timer()
from catboost import CatBoostClassifier
from matplotlib import pyplot
# evaluate the model
#cat_features_index = [0,1,2,3,4,5,6]
#,cat_features= cat_features_index
#model = CatBoostClassifier(verbose=0, n_estimators=100)
# fit the model on the whole dataset
model_cat = CatBoostClassifier(verbose=0, n_estimators=100)
model_cat.fit(X_train, y_train)
y_pred_cat = model_cat.predict(X_test)
y_pred_prob_cat = model_cat.predict_proba(X_test.values)[:,1]
stop = timeit.default_timer()
print(stop - start)


# In[46]:


y_pred_cat


# In[47]:


# xgboost for regression
start = timeit.default_timer()
from xgboost import XGBClassifier
# fit the model on the whole dataset
model_xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='mlogloss')
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
y_pred_prob_xgb = model_xgb.predict_proba(X_test.values)[:,1]
stop = timeit.default_timer()
print(stop - start)


# In[48]:


y_pred_xgb


# In[49]:


y_pred_prob_xgb


# In[50]:


from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingClassifier


# In[51]:


# make a prediction with a stacking mlxten
start = timeit.default_timer()

from sklearn.linear_model import LinearRegression
import mlxtend
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression

# define meta learner model
lr = LogisticRegression()
# define the stacking ensemble
model = StackingClassifier(classifiers=[model_lgb, model_cat, model_xgb], meta_classifier=lr)
# fit the model on all available data
model = model.fit(X_train, y_train)
#pred = list(pred.ravel())
#stack_result = list(stack_result.ravel())

stop = timeit.default_timer()
print(stop - start)


# In[52]:


stack_result = model.predict(X_test)
stack_result_prob = model.predict_proba(X_test.values)[:,1]


# In[53]:


stack_result_prob


# In[54]:


stack_result


# ## Checking the models Accuracy

# In[55]:


from sklearn import metrics
Lightgbm_accuracy = metrics.accuracy_score(y_test, y_pred_lgb)
Catboost_accuracy = metrics.accuracy_score(y_test, y_pred_cat)
Xgboost_accuracy = metrics.accuracy_score(y_test, y_pred_xgb)
Stacking_accuracy = metrics.accuracy_score(y_test, stack_result)


# In[56]:


Lightgbm_accuracy


# In[57]:


Catboost_accuracy


# In[58]:


Xgboost_accuracy


# In[59]:


Stacking_accuracy


# # Save Our Model
# * Serialization
# * Pickle
# * Joblib
# * numpy/json/ray

# In[60]:


# Using Joblib
import joblib

model_file = open("models/lgb_model_21.pkl","wb")
joblib.dump(model_lgb,model_file)
model_file.close()

model_file = open("models/cat_model_21.pkl","wb")
joblib.dump(model_cat,model_file)
model_file.close()

model_file = open("models/xgb_model_21.pkl","wb")
joblib.dump(model_xgb,model_file)
model_file.close()

model_file = open("models/stack_model_21.pkl","wb")
joblib.dump(model,model_file)
model_file.close()


# Prediction = pd.DataFrame({
#     'member_id' : member_id,
#     'loan_status' : stack_result_prob
# })
# Prediction = Prediction[['member_id','loan_status']]

# Prediction.to_csv('datasets/Prdeiction.csv', index = False)

# In[61]:


metrics.confusion_matrix(y_test, y_pred_xgb)


# In[62]:


dataset.shape


# In[63]:


X_test.shape


# In[64]:


dataset.head()


# In[65]:


dataset.isnull().sum()


# In[66]:


dataset.columns


# In[ ]:




