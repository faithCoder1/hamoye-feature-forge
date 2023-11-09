
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import category_encoders
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
import plotly.graph_objects as go

# Load the trained model
model = pickle.load(open('model_pkl', 'rb'))

# Streamlit UI
st.title('The Effect of Social Infrastructure and Education on Litracy Rate')

st.sidebar.header('To get the Literacy Rate , Select the values below')

st.write(
    """
    Literacy is the cornerstone of education and personal development. It is not merely the ability to read and write
    but a fundamental skill that empowers individuals to access knowledge, communicate effectively, and participate in the
    socio-economic development of society. Low literacy rates can have profound implications for a society's progress,
    and it is imperative to understand how various factors, including access to electricity, radio, television, primary and
    secondary school attendance, and improved sanitation and water, can impact literacy rates and educational infrastructure. 
    We'll explore the problems associated with low literacy rates and the features that can significantly influence literacy and education.
    """
)

# Input form
country_name = st.sidebar.selectbox('Country Name', [
    "Liberia", "Burundi", "Lesotho", "Zimbabwe", "Kenya", "Congo, Democratic Republic of", "Swaziland", "Namibia",
    "Rwanda", "Sao Tome and Principe", "Madagascar", "Tanzania", "Malawi", "Ethiopia", "Congo, Republic of",
    "Uganda", "Zambia", "Ghana", "Cameroon", "Mozambique", "Nigeria", "Guinea", "Sierra Leone", "Niger", "Mali", "Senegal", "Burkina Faso"
])

electricity = st.sidebar.slider('Electricity (%)', min_value=0, max_value=100)
radio = st.sidebar.slider('Radio (%)', min_value=0, max_value=100)
tv = st.sidebar.slider('TV (%)', min_value=0, max_value=100)
radio_tv = st.sidebar.slider('Radio and TV (%)', min_value=0, max_value=100)
pry_school = st.sidebar.slider('Primary School (%)', min_value=0, max_value=100)
sec_school = st.sidebar.slider('Secondary School (%)', min_value=0, max_value=100)
water = st.sidebar.slider('Water (%)', min_value=0, max_value=100)
sanitation = st.sidebar.slider('Sanitation (%)', min_value=0, max_value=100)


# Define a function to update the bar chart
def update_bar_chart(electricity, radio, tv, radio_tv, pry_school, sec_school, water, sanitation):
    # Create a Pandas DataFrame with the updated values
    input = {
        'Electricity': [electricity],
        'Radio': [radio],
        'Television': [tv],
        'Radio and/or Television': [radio_tv],
        'Net primary attendance rate (%)': [pry_school],
        'Net secondary attendance rate (%)': [sec_school],
        'Access to improved water ': [water],
        'Access to improved sanitation': [sanitation]
    }
    df1 = pd.DataFrame(input)

    # Create a bar chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df1.columns, y=df1.values[0]))

    # Set the y-axis range to be from 0 to 100
    fig.update_yaxes(range=[0, 100])

    # Customize x-axis appearance
    fig.update_xaxes(title_text='Features')

    # Display the updated Plotly bar chart using Plotly in Streamlit
    st.plotly_chart(fig)

# Update the bar chart based on slider values
update_bar_chart(electricity, radio, tv, radio_tv, pry_school, sec_school, water, sanitation)

# Create a dictionary to store input data
input_data = {
    'Country name': [country_name],
    'Electricity in household (% of population)': [electricity],
    'Radio in household (% of population)': [radio],
    'Television in household (% of population)': [tv],
    'Radio and/or Television in household (% of population)': [radio_tv],
    'Net primary attendance rate (%)': [pry_school],
    'Net secondary attendance rate (%)': [sec_school],
    'Access to improved water (% of population)': [water],
    'Access to improved sanitation (% of population)': [sanitation]
}


# Convert the dictionary to a pandas DataFrame
input_df = pd.DataFrame(input_data)

# Make the prediction using the model
prediction = model.predict(input_df)
literacy_rate = round(prediction[0], 2)

# Display the Literacy Rate within a styled card
st.markdown(
    f'<div style="background-color: #0074B8; padding: 20px; text-align: center; border-radius: 10px;">'
    f'<p style="color: white; font-size: 24px;">Literacy Rate</p>'
    f'<p style="color: white; font-size: 36px; font-weight: bold;">{literacy_rate:.2f}%</p>'
    f'</div>',
    unsafe_allow_html=True
)

st.write(f'Based on the data provided, your literacy rate is {literacy_rate}.format')