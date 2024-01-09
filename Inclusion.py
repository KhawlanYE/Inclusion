import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load data
df = pd.read_csv('Financial_inclusion_dataset.csv')

# Columns to encode
columns_to_encode = ['location_type', 'cellphone_access', 'gender_of_respondent', 'job_type', 'education_level',
                      'bank_account', 'country', 'marital_status']

# Encode categorical columns
le = LabelEncoder()
df[columns_to_encode] = df[columns_to_encode].apply(le.fit_transform)

# Features and target
x = df[['location_type', 'age_of_respondent', 'cellphone_access', 'education_level',
        'job_type', 'gender_of_respondent', 'household_size', 'year', 'country', 'marital_status']]
y = df['bank_account']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)

# Initialize and train the model
clf = RandomForestClassifier()
param_grid = {'n_estimators': [20, 30], 'max_depth': [10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)
grid = GridSearchCV(clf, param_grid, cv=5)
grid.fit(x_train_s, y_train)

# Streamlit app
st.title('Financial Inclusion in Africa app')
st.write('''**Dataset description**: The dataset contains demographic information and what financial services are used by approximately 33,600 
           individuals across East Africa. The ML model role is to predict which individuals are most likely to have or use a bank account.''')

# User inputs
user_inputs = {}
st.sidebar.header('User inputs parameters')
for col in x.columns:
    user_inputs[col] = st.sidebar.slider(f'{col}', float(x[col].min()), float(x[col].max()), float(x[col].mean()))

# Prediction button
if st.button('Predicting if the individual uses a bank account?'):
    user_data_scaled = scaler.transform(pd.DataFrame(user_inputs, index=[0]))
    prediction = grid.predict(user_data_scaled)
    prediction_mapping = {0: 'No', 1: 'Yes'}
    prediction_label = prediction_mapping.get(prediction[0], 'Unknown')
    st.success(f'Prediction: {prediction_label}')

df = pd.read_csv('Financial_inclusion_dataset.csv')
encoded_values_list = []
for column in columns_to_encode:
    original_values = df[column].unique()
    encoded_values = le.fit_transform(original_values)
    values_str = f"{', '.join([f'{orig} = {enc}' for orig, enc in zip(original_values, encoded_values)])}"
    encoded_values_list.append({'Column': column, 'Values': values_str})

st.subheader("User inputs Original and Encoded Values:")
st.table(pd.DataFrame(encoded_values_list).set_index('Column')['Values'])

