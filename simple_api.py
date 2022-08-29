import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template

app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def cancerPrediction():
	dataset_url = "datasets/breast_cancer.csv"
	df = pd.read_csv(dataset_url)

	
	inputQuery1=request.form['query1']
	inputQuery2=request.form['query2']
	inputQuery3=request.form['query3']
	inputQuery4=request.form['query4']
	inputQuery5=request.form['query5']

	df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

	train,test = train_test_split(df, test_size=0.2)
	features = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean','symmetry_mean']

	train_X = train[features]
	train_y = train.diagnosis

	test_X = test[features]
	test_y = test.diagnosis

	model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
	model.fit(train_X,train_y)

	prediction= model.predict(test_X)
	metrics.accuracy_score(test_y, prediction)


	data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]

	new_df = pd.DataFtame(data, columns = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean','symmetry_mean'])

	single = model.predict(new_df)
	probability = model.predict_proba(new_df)[:,1]

	if single==1:
		o1 = "The patient is diagnosed with Breast cancer"
		o2 = "Confidence: {}".format(probability*100)
	else:
		o1 = "The patient is not diagnosed with Breast cancer"
		o2 = "Confidence: {}".format(probability*100)


	return render_template('home.html',output1=o1, output2=o2, query1=request.form['query1'],query2=request.form['query2'],query3=request.form['query3'],query4=request.form['query4'],query5=request.form['query5'])

if __name__=="__main__":
    app.run(host="localhost", port=8000, debug=True)