import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import requests

# Load historical air quality data
def load_data():
    try:
        data = pd.read_excel('AirQualityUCI.xlsx')
        data.columns = data.columns.str.strip()

        print("Available columns:", data.columns.tolist())

        required_columns = ['Date', 'Time', 'CO(GT)', 'NO2(GT)']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            print(f"Missing columns in the dataset: {missing_columns}")
            return None, None

        # Drop unnecessary columns
        X = data.drop(columns=['Date', 'Time', 'CO(GT)'], errors='ignore')
        y = data[['CO(GT)', 'NO2(GT)']].dropna()

        if y.empty or X.empty:
            print("The feature or target variable dataframe is empty after dropping NaNs.")
            return None, None

        # Align X with y
        X = X.loc[y.index]
        return X, y

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse

# Make predictions
def predict(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# Fetch weather data from the API
def fetch_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching weather data: {response.status_code} - {response.text}")
        return None

# Main application function
def main():
    st.title('Air Quality Prediction App for Indian States')
    st.write("This app predicts multiple air quality indices based on real-time weather data.")

    # Load data and train model
    X, y = load_data()
    if X is None or y is None:
        st.error("Failed to load data. Please check the dataset.")
        return

    st.write(f"Number of features for training: {X.shape[1]}")  # Debug output

    model, mse = train_model(X, y)
    st.write(f'Model trained with Mean Squared Error: {mse:.2f}')

    city = st.text_input('Enter City Name')
    api_key = st.text_input("Enter your OpenWeatherMap API Key", type="password")

    if st.button('Get Weather and Predict'):
        if api_key and city:
            weather_data = fetch_weather_data(api_key, city)
            if weather_data:
                # Extract necessary features from weather data
                temperature = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                pressure = weather_data['main']['pressure']
                wind_speed = weather_data['wind']['speed']

                # Show weather information
                st.write(f"### Weather Information for {city}")
                st.write(f"- **Temperature**: {temperature} °C")
                st.write(f"- **Humidity**: {humidity}%")
                st.write(f"- **Pressure**: {pressure} hPa")
                st.write(f"- **Wind Speed**: {wind_speed} m/s")

                # Prepare input data for the model
                input_data = [
                    temperature,
                    humidity,
                    wind_speed,
                    pressure,
                    0,  # Placeholder for feature 5
                    0,  # Placeholder for feature 6
                    0,  # Placeholder for feature 7
                    0,  # Placeholder for feature 8
                    0,  # Placeholder for feature 9
                    0,  # Placeholder for feature 10
                    0,
                    0   # Placeholder for feature 11
                ]

                # Check if input_data has the correct number of features
                if len(input_data) == X.shape[1]:
                    predictions = predict(model, input_data)
                    st.success(f'The predicted Air Quality Indices for {city} are:\n' +
                               f'CO concentration: {predictions[0]:.2f} µg/m³\n' +
                               f'NO2 concentration: {predictions[1]:.2f} µg/m³')
                else:
                    st.error(f"Expected {X.shape[1]} features for prediction, but got {len(input_data)}.")
        else:
            st.warning("Please enter both API key and city name.")

if __name__ == "__main__":
    main()
    ## api KEY IS 3ab9ee862bddf524d11584d748277cfb