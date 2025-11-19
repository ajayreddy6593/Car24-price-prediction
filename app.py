import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load the saved Random Forest model
loaded_model = joblib.load('random_forest_model.pkl')

# Set the title of the Streamlit application
st.title('Used Car Price Prediction')

# Add a brief introductory text
st.write('Enter the car features to predict the price.')

try:
    unique_names = df_original['name'].unique().tolist()
    unique_makes = df_original['make'].unique().tolist()
    unique_models = df_original['model'].unique().tolist()
    unique_cities = df_original['city'].unique().tolist()
    unique_fueltypes = df_original['fueltype'].unique().tolist()
    unique_transmissions = df_original['transmission'].astype(str).unique().tolist() # Include NaN as string
    unique_bodytypes = df_original['bodytype'].astype(str).unique().tolist()     # Include NaN as string
    unique_registrationcities = df_original['registrationcity'].unique().tolist()
    unique_registrationstates = df_original['registrationstate'].unique().tolist()
except NameError:
    st.error("Could not access original data for unique values. Please ensure 'df_original' is loaded or provide unique values manually.")
    st.stop() # Stop the app if unique values are not available


input_name = st.selectbox('Car Name', unique_names)
input_make = st.selectbox('Make', unique_makes)
input_model = st.selectbox('Model', unique_models)
input_city = st.selectbox('City', unique_cities)
input_year = st.number_input('Year', min_value=2000, max_value=datetime.now().year + 1, value=2015)
input_kilometerdriven = st.number_input('Kilometers Driven', min_value=0, value=50000)
input_ownernumber = st.number_input('Owner Number', min_value=1, value=1)
input_fueltype = st.selectbox('Fuel Type', unique_fueltypes)
input_transmission = st.selectbox('Transmission', unique_transmissions)
input_bodytype = st.selectbox('Body Type', unique_bodytypes)
input_isc24assured = st.checkbox('Is Cars24 Assured?')
input_registrationcity = st.selectbox('Registration City', unique_registrationcities)
input_registrationstate = st.selectbox('Registration State', unique_registrationstates)
input_benefits = st.number_input('Benefits', min_value=0, value=0)
input_discountprice = st.number_input('Discount Price', min_value=0, value=0)

# Add a button to trigger the prediction
predict_button = st.button('Predict Price')

# Create a dictionary to store user inputs
user_input = {
    'name': input_name,
    'make': input_make,
    'model': input_model,
    'city': input_city,
    'year': input_year,
    'kilometerdriven': input_kilometerdriven,
    'ownernumber': input_ownernumber,
    'fueltype': input_fueltype,
    'transmission': input_transmission,
    'bodytype': input_bodytype,
    'isc24assured': input_isc24assured,
    'registrationcity': input_registrationcity,
    'registrationstate': input_registrationstate,
    'benefits': input_benefits,
    'discountprice': input_discountprice
}

# Preprocess the user inputs
# We need the label encoders used during training. Assuming 'le' is accessible or we can recreate it
# based on the unique values observed during training.
# Let's recreate the label encoder mappings based on the original df for robustness.

label_encoders = {}
categorical_cols = ['name', 'make', 'model', 'city', 'fueltype', 'transmission', 'bodytype', 'registrationcity', 'registrationstate']
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on original data's unique values, handling potential NaN by converting to string
    le.fit(df_original[col].astype(str).unique())
    label_encoders[col] = le

# Apply label encoding to user inputs
preprocessed_input = {}
for col, value in user_input.items():
    if col in categorical_cols:
        # Handle potential new values not seen during training by assigning a default value
        # Here, we'll assign -1 or a value outside the fitted range.
        try:
            # Convert input value to string before transforming
            preprocessed_input[col] = label_encoders[col].transform([str(value)])[0]
        except ValueError:
            preprocessed_input[col] = -1 # Assign a default value for unseen categories
    else:
        preprocessed_input[col] = value

# Calculate derived features
current_year = datetime.now().year
preprocessed_input['car_age'] = current_year - preprocessed_input['year']

# Handle division by zero for price_per_km
# Use a placeholder price, e.g., the mean price from the training data (if available)
# In a real app.py, you would load the mean price or handle this based on domain knowledge.
try:
    # Assuming df is available with the 'price' column after cleaning
    placeholder_price = df['price'].mean()
except NameError:
    placeholder_price = 500000 # Fallback placeholder price if df is not available


if preprocessed_input['kilometerdriven'] == 0:
    preprocessed_input['price_per_km'] = 1e6  # Assign a large number for 0 km driven
else:
    preprocessed_input['price_per_km'] = placeholder_price / preprocessed_input['kilometerdriven']


preprocessed_input['is_1st_owner'] = 1 if preprocessed_input['ownernumber'] == 1 else 0

# Use the label encoded value for transmission to determine is_automatic
try:
    # Convert 'Automatic' to its label encoded value
    automatic_transmission_label = label_encoders['transmission'].transform(['Automatic'])[0]
    # Compare the preprocessed input value to the label encoded 'Automatic' value
    preprocessed_input['is_automatic'] = 1 if preprocessed_input['transmission'] == automatic_transmission_label else 0
except ValueError:
    preprocessed_input['is_automatic'] = 0 # Default to not automatic if 'Automatic' was not in training data


# Convert boolean isc24assured to integer
preprocessed_input['isc24assured'] = int(preprocessed_input['isc24assured'])

# Convert the preprocessed input dictionary to a DataFrame
input_data = pd.DataFrame([preprocessed_input])

# Ensure the columns are in the same order as X_train and include all necessary columns
# We need to make sure all columns in X_train are present, even if their input comes from derived features.
# Create a list of columns from X_train, excluding 'log_price' if it exists
try:
    # Assuming X_train is available from the notebook state
    training_columns = [col for col in X_train.columns if col != 'log_price']
except NameError:
    st.error("Could not access training columns. Please ensure 'X_train' is available.")
    st.stop() # Stop the app if training columns are not available


# Reindex the input_data DataFrame to match the training columns order
input_data = input_data[training_columns]


# Make predictions
if predict_button:
    # Make a prediction using the loaded model
    predicted_price = loaded_model.predict(input_data)

    # Display the predicted price
    st.subheader("Predicted Car Price:")
    # Format the output nicely, e.g., in Indian Rupees
    st.write(f"₹ {predicted_price[0]:,.2f}")