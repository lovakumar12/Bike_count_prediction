from src.mlProject.pipeline.prediction import PredictionPipeline
from src.mlProject.pipeline.training import TrainingPipeline
import streamlit as st

def main():
    st.title("🚲 Bike Rental Prediction System")
    
    if st.sidebar.button("Train Model"):
        training_pipeline = TrainingPipeline()
        training_pipeline.start_training()
        st.success("Model training completed!")
    
    prediction_pipeline = PredictionPipeline()
    
    # Input parameters
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
    
    if st.button("Predict"):
        features = {
            'Hour': hour,
            'Temperature(°C)': temperature,
            'Humidity(%)': humidity,
            'Wind speed (m/s)': wind_speed,
            'Visibility (10m)': visibility,
            'Dew point temperature(°C)': dew_point,
            'Solar Radiation (MJ/m2)': solar_radiation,
            'Rainfall(mm)': rainfall,
            'Snowfall (cm)': snowfall,
            'Seasons': season,
            'Holiday': holiday,
            'Functioning Day': functioning
        }
        
        prediction = prediction_pipeline.predict(features)
        st.success(f"Predicted Bike Rentals: {int(prediction[0])}")

if __name__ == "__main__":
    main()