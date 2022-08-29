# prediction function
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
application = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    flt_features = list(map(float, to_predict_list))
    final_features = [np.array(flt_features)]
    prediction = model.predict(final_features)
    prediction = model.predict_proba(final_features)
    output  = prediction[0][0]
#    output = round(prediction[0], 2) 
    old_value = output
    old_min = 0
    old_max = 1
    new_min = 300
    new_max = 850
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    if new_value < 580:
        perfomance = "Poor Score"
    elif 580 <= new_value < 670:
        perfomance = "Fair Score"
    elif 670 <= new_value < 740:
        perfomance = "Good Score"
    elif 740 <= new_value < 800:
        perfomance = "Very Good Score"
    else:
        perfomance = "Exceptional"

    return render_template('index.html', prediction_text='The probability of not defaulting the loan is  {:.2f} .The credit score of this customer is {:.2f} Grade: {}'.format(output,new_value, perfomance))




@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    application.run(debug=True)