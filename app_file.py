import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved model
loaded_model = pickle.load(open('trained_model3.sav', 'rb'))

# Load LabelEncoders used during training
label_encoders = pickle.load(open('label_encoders.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))
target_encoder = pickle.load(open('target_encoder.sav', 'rb'))

# Define a function to preprocess input data
def preprocess_input(input_data):
    input_data_processed = input_data.copy()
    
    # Apply the saved LabelEncoders to the input data
    for col in label_encoders:
        input_data_processed[col] = label_encoders[col].transform([input_data[col]])[0]

    # Convert input_data to numpy array
    input_data_as_numpy_array = np.asarray(list(input_data_processed.values()))

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Apply the saved StandardScaler
    input_data_scaled = scaler.transform(input_data_reshaped)
    
    return input_data_scaled

# Streamlit web app code
def main():
    st.title("Suicidal thoughts Prediction App")

    # Initialize session state for input data and prediction result
    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = {}
        st.session_state['prediction'] = None

    # User input for each feature
    input_features = ['Age', 'Gender', 'Education', 'Live with',
                      'Conflict with law', 'Most used drugs', 'Motive about drug',
                      'motivation by friends', 'Spend most time',
                      'Mental/emotional problem', 'Family relationship',
                      'Financials of family', 'Addicted person in family', 
                      'no. of friends', 'Withdrawal symptoms', 
                      'Satisfied with workplace', 'Case in court',
                      'Living with drug user', 'Smoking',
                      'Easy to control use of drug', 'Frequency of drug usage',
                      'Taken drug while experiencing stress']

    for feature in input_features:
        st.session_state['input_data'][feature] = st.text_input(
            feature, 
            st.session_state['input_data'].get(feature, "")
        )

    # Preprocess input data
    if st.button("Predict"):
        input_data_processed = preprocess_input(st.session_state['input_data'])
        
        # Make prediction
        prediction = loaded_model.predict(input_data_processed)
        st.session_state['prediction'] = 'no' if prediction[0] == 0 else 'yes'
    
    # Show the prediction result if available
    if st.session_state['prediction'] is not None:
        st.write(f"The prediction is '{st.session_state['prediction']}'")

if __name__ == "__main__":
    main()
