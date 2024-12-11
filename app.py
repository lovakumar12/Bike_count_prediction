import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Bike Rental Prediction", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('SeoulBikeData.csv',encoding = 'unicode_escape')
    return df

def preprocess_data(df):
    # Create label encoders
    le_season = LabelEncoder()
    le_holiday = LabelEncoder()
    le_functioning = LabelEncoder()
    
    # Encode categorical variables
    df['Seasons_encoded'] = le_season.fit_transform(df['Seasons'])
    df['Holiday_encoded'] = le_holiday.fit_transform(df['Holiday'])
    df['Functioning_Day_encoded'] = le_functioning.fit_transform(df['Functioning Day'])
    
    return df, le_season, le_holiday, le_functioning

# Load data
df = load_data()
df, le_season, le_holiday, le_functioning = preprocess_data(df)

# Feature selection
features = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
           'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)',
           'Rainfall(mm)', 'Snowfall (cm)', 'Seasons_encoded', 'Holiday_encoded', 
           'Functioning_Day_encoded']

X = df[features]
y = df['Rented Bike Count']

# Train model
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = train_model()

# Streamlit UI
st.title("🚲 Bike Rental Prediction System")

# Sidebar for user input
st.sidebar.header("Input Parameters")

hour = st.sidebar.slider("Hour", 0, 23, 12)
temperature = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 20.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)
visibility = st.sidebar.slider("Visibility (10m)", 0, 2000, 1000)
dew_point = st.sidebar.slider("Dew Point Temperature (°C)", -30.0, 30.0, 0.0)
solar_radiation = st.sidebar.slider("Solar Radiation (MJ/m2)", 0.0, 5.0, 1.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
snowfall = st.sidebar.slider("Snowfall (cm)", 0.0, 30.0, 0.0)
season = st.sidebar.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Autumn'])
holiday = st.sidebar.selectbox("Holiday", ['No Holiday', 'Holiday'])
functioning = st.sidebar.selectbox("Functioning Day", ['Yes', 'No'])

# Make prediction
def predict():
    input_data = np.array([[
        hour, temperature, humidity, wind_speed, visibility, dew_point,
        solar_radiation, rainfall, snowfall,
        le_season.transform([season])[0],
        le_holiday.transform([holiday])[0],
        le_functioning.transform([functioning])[0]
    ]])
    
    prediction = model.predict(input_data)
    return max(0, int(prediction[0]))

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Predictions")
    if st.button("Predict"):
        prediction = predict()
        st.success(f"Predicted Bike Rentals: {prediction}")
    
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title("Feature Importance")
    st.pyplot(fig)

with col2:
    st.subheader("Data Overview")
    st.write("Sample of the dataset:")
    st.dataframe(df.head())
    
    st.subheader("Temperature vs Rentals")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Temperature(°C)', y='Rented Bike Count')
    plt.title("Temperature vs Bike Rentals")
    st.pyplot(fig)

# Model performance metrics
st.subheader("Model Performance")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

col3, col4 = st.columns(2)
with col3:
    st.metric("Training R² Score", f"{train_score:.3f}")
with col4:
    st.metric("Testing R² Score", f"{test_score:.3f}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Random Forest Regressor")