import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app=Flask(__name__,template_folder='templates')
model = pickle.load(open("cat_model_pickle.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('pred.html')

@app.route('/predict',methods=['POST'])   
def predict():
    '''
    For rendering results on HTML GUI
    '''
    employee_id = request.form("Employee ID")
    date_joining = request.form["Date of Joining"]
    gender =request.form["Gender"]
    company=request.form["Company Type"]
    wfh =request.form["WFH Setup Available"]	
    designation=request.form["Designation"]	
    hours =request.form["Resource Allocation"]
    fatigue =request.form["Mental Fatigue Score"]


   ### id = 
  ####  list_features = [ x for x in request.form.values()]
 ##   final_features = [np.list(int_features)]
  #  prediction = model.predict(final_features) 
    
  #  output = round(prediction[0], 2)

  #  return render_template('pred.html', prediction_text='Employee burnout: {}'.format(output))  ###

    prediction = model.predict([employee_id,date_joining, gender,company, wfh, designation,hours,fatigue])

    output=round(prediction[0],2)
    return render_template('pred.html',prediction_text='Employee burnout: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)