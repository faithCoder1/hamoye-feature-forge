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
    int_features = [float(x) for x in request.form.values()]
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
        'Country name': country_name,
        'Electricity in household (% of population)': electricity,
        'Radio in household (% of population)': radio,
        'Television in household (% of population)': tv,
        'Radio and/or Television in household (% of population)': radio_tv,
        'Net primary attendance rate (%)': pry_school,
        'Net secondary attendance rate (%)': sec_school,
        'Access to improved water (% of population)': water,
        'Access to improved sanitation (% of population)': sanitation
    }

    # Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame(input_data, index=[0])
    print(input_df)

    # Make the prediction using the model
    prediction = model.predict(input_df)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ =="__main__":
    app.run(debug=True)
