import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('LogisticRegression.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Titanic Survival Prediction")

st.write("This app predicts whether a passenger would survive the Titanic disaster based on their details.")

# Input features
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Predict button
if st.button("Predict Survival"):
    # Encode 'Sex': male=0, female=1 (as used during training)
    sex_encoded = 1 if sex == 'female' else 0

    # Encode 'Embarked': S=0, C=1, Q=2 (assuming this was your mapping)
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    embarked_encoded = embarked_map[embarked]

    # Create a DataFrame with the encoded inputs
    user_input = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked_encoded]
    })

    # Predict
    prediction = model.predict(user_input)[0]

    if prediction == 1:
        st.success("The passenger would have SURVIVED!")
    else:
        st.error("The passenger would NOT have survived.")
