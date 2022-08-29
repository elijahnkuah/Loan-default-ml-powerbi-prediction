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
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer as Imputer
from sklearn import preprocessing
import re
import timeit
import random
import joblib

random.seed(3)

class data_processing():
    dataset = pd.read_csv("datasets/dataset.csv")
    ''' After importing the dataset, you need to drop unnecessary variables'''
    
    dataset = dataset.drop(['member_id','total_rec_int','batch_enrolled','funded_amnt_inv','title','grade', 'sub_grade','desc', 'zip_code','addr_state','emp_title','verification_status', 'pymnt_plan','delinq_2yrs',
       'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record','open_acc', 'pub_rec','initial_list_status','total_rec_late_fee',
       'recoveries', 'collection_recovery_fee','tot_coll_amt', 'collections_12_mths_ex_med', 'mths_since_last_major_derog','verification_status_joint', 'last_week_pay','acc_now_delinq','loan_status'], axis=1)
    
    y = dataset.loan_status.values

    '''Converting all strings variables to floats and integers'''
    dataset['emp_length'] = dataset['emp_length'].replace({np.nan:0, '< 1 year':1, '1 year':2, '2 years':3, '3 years':4, '4 years':5, '5 years':6, '6 years':7, '7 years':8, '8 years':9, '9 years':10, '10+ years':11})
    dataset['term'] = dataset['term'].replace({'36 months':36, '60 months':60})
    dataset['home_ownership']=dataset['home_ownership'].replace({'OWN':1, 'MORTGAGE':2, 'RENT':3, 'OTHER':4, 'NONE':5, 'ANY':6})
    dataset['purpose'] = dataset['purpose'].replace({'small_business':1,'debt_consolidation':2, 'home_improvement':3, 'credit_card':4,'major_purchase':5,  
                                                     'vacation':6, 'car':7, 'moving':8,'medical':9, 'wedding':10, 'renewable_energy':11, 'house':12, 'educational':13,'other':14,})
    dataset['application_type'] = dataset['application_type'].replace({'INDIVIDUAL':1, 'JOINT':2})
    
    
    
    value = -9999
    def helping_features(value):
        '''Creating new variables'''
        i = ((dataset['int_rate'])/100)/12
        d = 1 - (1/(1+i)**dataset['term'])
        dataset['Monthly_supposed_payment'] = (dataset['funded_amnt']*i)/d
        dataset['Total_refund'] = dataset['Monthly_supposed_payment']*dataset['term']
        dataset['Interest_amnt'] = dataset['Total_refund'] - dataset['funded_amnt']
        dataset['Monthly_income'] = dataset['annual_inc'].apply(lambda x : x/12 if x >=0 else -9999)
    helping_features(-9999)
    
    def fill_nulls(value):
        '''filling missing values'''
        cols_fill = ['annual_inc','revol_util', 'total_acc', 'tot_cur_bal', 'total_rev_hi_lim']
        
        if value == -9999:
            for col in cols_fill:
                dataset.loc[dataset[col].isnull(), col] = -9999
        else : 
            for col in cols_fill:
                dataset.loc[dataset[col].isnull(), col] = dataset[col].median()
    fill_nulls(-9999)
    
    '''Scalling the Dataset'''
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
    from sklearn.feature_selection import SelectKBest, f_classif
    df = SelectKBest(f_classif, k=3)
    df1 = StandardScaler()
    data_scale = df1.fit_transform(dataset) 
    data = data_scale
    
    columns = dataset.columns
    
    data_df = pd.DataFrame(data, columns=columns)
    data = data_df
    data.head()

    '''Splitting the data '''
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(data,y, test_size=0.2, random_state=0)

    '''Training The data'''
    # fit the model on the whole dataset
    model_lgb = LGBMClassifier()
    model_lgb.fit(X_train, y_train)
    y_pred_lgb = model_lgb.predict(X_test)
    y_pred_prob_lgb = model_lgb.predict_proba(X_test.values)[:,1]

    # Save model

    model_file = open("models/lgb_model_21.pkl","wb")
    joblib.dump(model_lgb,model_file)
    model_file.close()