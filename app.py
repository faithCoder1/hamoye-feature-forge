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

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Collect the input data from the form
    country_name = request.form['Country_name']
    electricity = float(request.form['Electricity'])
    radio = float(request.form['Radio'])
    tv = float(request.form['TV'])
    radio_tv = float(request.form['Radio_Tv'])
    pry_school = float(request.form['pry_School'])
    sec_school = float(request.form['Sec_School'])
    water = float(request.form['Water'])
    sanitation = float(request.form['Sanitation'])

    # Create a dictionary to store all input data
    input_data = {
        'Country_name': country_name,
        'Electricity': electricity,
        'Radio': radio,
        'TV': tv,
        'Radio_Tv': radio_tv,
        'pry_School': pry_school,
        'Sec_School': sec_school,
        'Water': water,
        'Sanitation': sanitation
    }

    # Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Make the prediction using the model
    prediction = model.predict(input_df)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ =="__name_":
    app.run(debug=True)

