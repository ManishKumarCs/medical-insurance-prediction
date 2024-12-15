import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the Random Forest model (previously trained)
rf_model = pkl.load(open('MIPML.pkl', 'rb'))

# Load dataset for Linear Regression training
data = pd.read_csv("insurance.csv")
data.replace({'sex':{'female':0, 'male':1}}, inplace=True)
data.replace({'smoker':{'no':0, 'yes':1}}, inplace=True)
data.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)

# Prepare the data for Linear Regression model
X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Streamlit app
st.title('🩺 Medical Insurance Cost Prediction')

# Sidebar for inputs
st.sidebar.header('Input Details')
age = st.sidebar.slider('Age', min_value=18, max_value=80, value=30)
bmi = st.sidebar.slider('BMI (Body Mass Index)', min_value=10, max_value=50, value=22)
children = st.sidebar.selectbox('Number of Children', list(range(6)))

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox('Gender', ['Female', 'Male'])
with col2:
    smoker = st.selectbox('Smoker?', ['No', 'Yes'])
with col3:
    region = st.selectbox('Region', ['SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'])

# Convert categorical inputs to numerical
gender = 0 if gender == 'Female' else 1
smoker = 1 if smoker == 'Yes' else 0
region_dict = {'SouthEast': 0, 'SouthWest': 1, 'NorthEast': 2, 'NorthWest': 3}
region = region_dict[region]

# Prepare input data array for prediction
input_data = (age, gender, bmi, children, smoker, region)
input_data_array = np.asarray(input_data).reshape(1, -1)

# Add Predict Cost button
if st.button('🔮 Predict Cost'):
    # Make predictions using both models
    predicted_premium_rf = rf_model.predict(input_data_array)[0]
    predicted_premium_lr = linear_model.predict(input_data_array)[0]

    # Display the predictions
    st.success(f'💵 Estimated Insurance Cost (Random Forest): **${predicted_premium_rf:.2f} USD**')
    st.success(f'💵 Estimated Insurance Cost (Linear Regression): **${predicted_premium_lr:.2f} USD**')

    # ---- Plot comparison of predictions vs age for both models ----
    ages = list(range(18, 81))
    predictions_rf = []
    predictions_lr = []

    for test_age in ages:
        test_input = np.asarray([test_age, gender, bmi, children, smoker, region]).reshape(1, -1)
        pred_rf = rf_model.predict(test_input)[0]
        pred_lr = linear_model.predict(test_input)[0]
        predictions_rf.append(pred_rf)
        predictions_lr.append(pred_lr)

    # Line chart to compare both models' predictions vs age
    age_df = pd.DataFrame({
        'Age': ages,
        'Random Forest Prediction': predictions_rf,
        'Linear Regression Prediction': predictions_lr
    })

    # Create line chart using Plotly
    fig = px.line(age_df, x='Age', y=['Random Forest Prediction', 'Linear Regression Prediction'],
                  title='Comparison of Insurance Cost Predictions (Random Forest vs Linear Regression)',
                  labels={'value': 'Insurance Cost (USD)', 'Age': 'Age'},
                  height=400)

    # Show the chart
    st.plotly_chart(fig)

    # Explanation of the graph
    st.markdown("""
    ### Graph Explanation
    This graph displays the estimated insurance costs predicted by two different models: **Random Forest** and **Linear Regression**.

    - **X-Axis (Age)**: The age of the individual, ranging from 18 to 80 years.
    - **Y-Axis (Insurance Cost)**: The estimated insurance costs in USD.

    **Key Points**:
    - The **Random Forest model** generally captures complex relationships and may provide more accurate predictions for various age groups compared to the **Linear Regression model**, which assumes a linear relationship between age and insurance cost.
    - As age increases, you may notice trends where the predicted costs change, reflecting the insurance company's pricing strategies.
    - Compare the two lines to see how they differ in their predictions; larger gaps could indicate where one model might be more accurate than the other.

    Use this information to understand how age and other factors can influence insurance premiums.
    """)

