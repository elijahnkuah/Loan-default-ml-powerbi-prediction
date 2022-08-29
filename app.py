# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:00:38 2021

@author: Elijah_Nkuah
"""
# Importing needful libraries
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from PIL import Image


# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib
matplotlib.use('Agg')

# import Database
from database_loan import *

# Function for password to connect with the Signup
def generate_pass(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def verify_pass(password,pass_text):
	if generate_pass(password) == pass_text:
		return pass_text
	return False


feature_names_best = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'emp_length',
       'home_ownership', 'annual_inc', 'purpose', 'dti', 'revol_bal',
       'revol_util', 'total_acc', 'application_type', 'tot_cur_bal',
       'total_rev_hi_lim', 'Monthly_supposed_payment', 'Total_refund','Interest_amnt', 'Monthly_income']


term_dict = {'36 months':36, '60 months':60}
home_ownership_dict = {'OWN':1, 'MORTGAGE':2, 'RENT':3, 'OTHER':4, 'NONE':5, 'ANY':6}
purpose_dict = {'small_business':1,'debt_consolidation':2, 'home_improvement':3, 'credit_card':4,'major_purchase':5,  
           'vacation':6, 'car':7, 'moving':8,'medical':9, 'wedding':10, 'renewable_energy':11, 'house':12, 'educational':13,'other':14}
application_type_dict = {'INDIVIDUAL':1, 'JOINT':2}
emp_length_dict = {str(np.nan):0, '< 1 year':1, '1 year':2, '2 years':3, '3 years':4, '4 years':5, '5 years':6, '6 years':7, '7 years':8, '8 years':9, '9 years':10, '10+ years':11}    

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key
def get_tvalue(val):
	term_dict = {'36 months':36, '60 months':60}
	for key,value in term_dict.items():
		if val == key:
			return value
def get_hvalue(val):
	home_ownership_dict = {'OWN':1, 'MORTGAGE':2, 'RENT':3, 'OTHER':4, 'NONE':5, 'ANY':6}
	for key,value in home_ownership_dict.items():
		if val == key:
			return value
def get_pvalue(val):
	purpose_dict = {'small_business':1,'debt_consolidation':2, 'home_improvement':3, 'credit_card':4,'major_purchase':5,  
           'vacation':6, 'car':7, 'moving':8,'medical':9, 'wedding':10, 'renewable_energy':11, 'house':12, 'educational':13,'other':14}
	for key,value in purpose_dict.items():
		if val == key:
			return value
def get_avalue(val):
    application_type_dict = {'INDIVIDUAL':1, 'JOINT':2}
    for key,value in application_type_dict.items():
            if val == key:
                    return value
def get_evalue(val):
    emp_length_dict = {np.nan:0, '< 1 year':1, '1 year':2, '2 years':3, '3 years':4, '4 years':5, '5 years':6, '6 years':7, '7 years':8, '8 years':9, '9 years':10, '10+ years':11}
    for key,value in emp_length_dict.items():
            if val == key:
                    return value

# Function to load machine learning Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# Machine learning Interpretation

import lime
import lime.lime_tabular


html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Reducing Risk of a Bank by predicting a customer who can default loan </h1>
		<h4 style="color:white;text-align:center;">BANK LOAN DEFAULT </h4>
		</div>
		"""

# Avatar Image
avatar1 ="https://www.w3schools.com/howto/img_avatar1.png"
avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"

result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Customer {}s</p>
	</div>
	"""

result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Customer {}s</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""
Steps_to_follow ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Follow The Steps Below:</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Signup if not already having account from the sidebar</li>
		<li style="text-align:justify;color:black;padding:10px">After signing up;</li>
		<li style="text-align:justify;color:black;padding:10px">Log in with your details</li>
		<li style="text-align:justify;color:black;padding:10px">i.e Username & Password</li>
		<ul>
	</div>
	"""


descriptive_message_temp ="""
	<div style="background-color:silver;"overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:#FFA500;padding:10px">Problem Statement</h3>
		<p>The Bank Indessa has not done well in last 3 quarters. Their NPAs (Non Performing Assets) have reached all time high. It is starting to lose confidence of its investors. As a result, it’s stock has fallen by 20% in the previous quarter alone.
After careful analysis, it was found that the majority of NPA was contributed by loan defaulters. With the messy data collected over all the years, this bank has decided to use machine learning to figure out a way to find these defaulters and devise a plan to reduce them.</p>
	</div>
	"""
#background-color:#00008B;
@st.cache
def load_image(img):
	im =Image.open(os.path.join(img))
	return im

st.set_option('deprecation.showPyplotGlobalUse', True)

def change_avatar(sex):
	if sex == "male":
		avatar_img = 'img_avatar.png'
	else:
		avatar_img = 'img_avatar2.png'
	return avatar_img


def main():
	"""Prediction App for loan defaulters"""
	st.image(load_image('images/loan1.png'))
	menu = ["Home","Login","Signup"]
	submenu = ["Plot","Prediction","Metrics"]
	
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.header("Home")
		st.markdown(descriptive_message_temp,unsafe_allow_html=True)
		st.markdown(Steps_to_follow, unsafe_allow_html=True)
		st.sidebar.image(load_image('images/LOGO.png'))
        
	elif choice == "Login":
		st.sidebar.image(load_image('images/LOGO.png'))
		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = generate_pass(password)
			result = login_user(username,verify_pass(password,hashed_pswd))
			# if password == "12345":
			if result:
				st.success("Welcome {} to Loan Defaulters' Prediction App".format(username))
				st.image(load_image('images/welcome.png'))
				activity = st.selectbox("Activity",submenu)
				
				if activity == "Plot":
					st.subheader("Data Visualisation")
					df = pd.read_csv("datasets/dataset.csv")
					st.markdown("Total dataset used is {}".format(df.shape))
					st.write(df.head())
					fig, ax = plt.subplots()
					df['loan_status'].value_counts().plot(kind='bar', color="#ADD8E6", title="Loan paid:0, loan defaulted: 1")
					st.pyplot(fig)
					fig1, ax = plt.subplots()
					df['home_ownership'].value_counts().plot(kind='bar', title="Home Ownership",color="#D2691E")
					st.pyplot(fig1)
					fig2, ax = plt.subplots()
					df['purpose'].value_counts().plot(kind='bar', title="Purpose for the Loan")
					st.pyplot(fig2)
					fig3 = plt.figure(figsize=(15, 8))
					plt.title("The number of years customers have been working", fontsize=20)
					sns.countplot(x = "emp_length", data = df)
					st.pyplot(fig3)
					fig4 = plt.figure(figsize=(15, 8))
					plt.title("The number of months customers must completely finish payment", fontsize=20)
					sns.countplot(x = "term", data = df)
					st.pyplot(fig4)
				elif activity == "Prediction":
					st.subheader("Predictive Analytics")
					st.info("Not all dependent variables are needed for this particular App since we are now checking a new borrower default probability")
					
					loan_request  = st.number_input("Loan Amount requested by Customer",1,1000000000)
					funded_amnt = st.number_input("Amount the bank wish to give",1,1000000000)
					term = st.selectbox("Required months to pay the amount",tuple(term_dict.keys()))
					Interest_rate = st.number_input("Interest Rate",0.0,100.0)
					emp_length = st.selectbox("How many years has customer been working",tuple(emp_length_dict.keys()))
					home_owner = st.selectbox("Home Ownership",tuple(home_ownership_dict.keys()))
					annual_inc = st.number_input("What is Customer's annual income?", 0.0,1000000000.0)
					purpose = st.selectbox("What exactly is customer using the loan for? ", tuple(purpose_dict.keys()))
					dti = st.number_input("What is Customer's debt to income ratio? Example: if customer pays GHS2000 as monthly loan payment already and his/her gross income is GHS6000, then DTI=(2000/6000)*100% = 33.33",1.0,100.0)
					revol_bal = st.number_input("Total credit revolving balance",0.0,1000000000.0)
					revol_util = st.number_input("Credit Utilization Ratio", 0.0,100.0)
					total_acc = st.number_input("The total number of credit lines currently in the borrower's credit file.", 0,200)
					application_type = st.selectbox("Application Type",tuple(application_type_dict.keys()))
					tot_cur_bal = st.number_input("Total current balance of all accounts", 1.0, 1000000000.0)
					total_rev_hi_lim = st.number_input("Total revolving high credit/credit limit",1.0,10000000000.0)
					Monthly_supposed_payment = st.number_input("The amount customer must pay every month",1.0,10000000000.0)
					Total_refund = st.number_input("Total Amount to be paid (Required months to pay the amount * The amount customer must pay every month)", 1.0,10000000000.0)
					Interest_amnt = st.number_input("The total Interest Amount to be paid",1.0,10000000000.0)
					income = st.number_input("Customer's gross monthly income",1.0,10000000000.0)
					# Using the various functions to turn the inputs into numbers

					feature_list = [loan_request,funded_amnt,get_tvalue(term),Interest_rate,get_evalue(emp_length),get_hvalue(home_owner),annual_inc,get_pvalue(purpose),dti,revol_bal,revol_util,total_acc,get_avalue(application_type),tot_cur_bal,total_rev_hi_lim,Monthly_supposed_payment,Total_refund,Interest_amnt,income]
					st.write("The Number of independent varaiables is {}".format(len(feature_list)))
					#st.write(feature_list)
					pretty_result = {"loan_request":loan_request,"funded_amnt":funded_amnt,"term":term,"Interest_rate":Interest_rate,"emp_length":emp_length,"home_ownership":home_owner,"annual_inc":annual_inc,"purpose":purpose,"DTI":dti,"revol_bal":revol_bal,"revol_util":revol_util,"total_acc":total_acc,"application_type":application_type,"tot_cur_bal":tot_cur_bal,"total_rev_hi_lim":total_rev_hi_lim,"Monthly_supposed_payment":Monthly_supposed_payment,"Total_refund":Total_refund,'Interest_amnt':Interest_amnt,'Monthly Gross Income':income}
					st.json(pretty_result)
					single_sample = np.array(feature_list).reshape(1,-1)

					# Machine Learning models
					model_choice = st.selectbox("Select Model",["Catboost","Lightgbm","Xgboost"])
					if st.button("Predict"):
						if model_choice == "Catboost":
							loaded_model = load_model("models/cat_model_21.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						elif model_choice == "Lightgbm":
							loaded_model = load_model("models/lgb_model_21.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						else:
							st.info("This model's prediction is the other way round. Trying to debug and fix the issue. Thank you")
							loaded_model = load_model("models/xgb_model_21.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
							
						if prediction == 1:
							st.warning("This customer can default the loan")
							pred_probability_score = {"Probability of not defaulting the loan is ":pred_prob[0][0]*100,"Probability of defaulting the loan":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							
						elif prediction == 2:
							st.success("This customer can default the loan")
							pred_probability_score = {"Probability of not defaulting the loan is ":pred_prob[0][0]*100,"Probability of defaulting the loan":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
						
                            
						elif prediction == 3:
							st.success("This customer can default the loan")
							pred_probability_score = {"Probability of not defaulting the loan is ":pred_prob[0][0]*100,"Probability of defaulting the loan":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							
						else:
							st.success("This customer might not default the loan")
							pred_probability_score = {"Probability of not defaulting the loan is ":pred_prob[0][0]*100,"Probability of defaulting the loan":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)

			else:
				st.warning("Incorrect Username/Password")


	elif choice == "Signup":
		st.sidebar.image(load_image('images/LOGO.png'))
		new_username = st.text_input("User Name")
		new_password = st.text_input("Password", type='password')
		confirmed_password = st.text_input("Confirm Password", type='password')
		if new_password == confirmed_password:
			st.success("Password Confirmed")
		else:
			st.warning("Passwords not the same")
		if st.button("Submit"):
			create_usertable()
			hashed_new_password = generate_pass(new_password)
			add_userdata(new_username, hashed_new_password)
			st.success("You have successfully created a new account")
			st.info("Login To Get Started")


if __name__ == '__main__':
	main()