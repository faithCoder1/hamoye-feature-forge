import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model_pkl','rb')) ## loading the trained model

@app.route('/') ##HTML home page
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.value()]
    feature = [np.array(int_features)]
    prediction = model.predict(feature)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text = '{}'.format(output))

if __name__ =="__name_":
    app.run(debug=True)

