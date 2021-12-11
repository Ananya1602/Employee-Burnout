import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("cat_model_pickle.pkl", 'rb'))

@app.route('/')
def home_page():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predict_page():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('predict.html', prediction_text='Employee Burnout Score is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)