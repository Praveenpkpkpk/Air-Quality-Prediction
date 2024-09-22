import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load and preprocess the dataset
def load_data():
    data = pd.read_excel('C:/Users/prave/OneDrive/Documents/Desktop/Air Quality/AirQualityUCI.xlsx')
    data.columns = data.columns.str.strip()

    # Drop unnecessary columns and define features and target
    X = data.drop(columns=['Date', 'Time', 'CO(GT)'])  # 'CO(GT)' is the target column
    y = data['CO(GT)']  # Target variable
    
    X = X.dropna()
    y = y[X.index]  # Ensure target matches the input rows
    
    return X, y

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse

# Prediction function
def predict(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# Simulated air quality sensor data for Indian states
state_sensor_data = {
    'Andhra Pradesh': [1600, 300, 4.5, 1800, 140, 1300, 45, 1100, 1000, 29.0, 60.0, 20.0],
    'Arunachal Pradesh': [1400, 100, 2.0, 1200, 60, 950, 20, 900, 850, 15.0, 55.0, 10.0],
    'Assam': [1500, 200, 3.0, 1300, 80, 1100, 30, 1000, 900, 20.0, 50.0, 15.0],
    'Bihar': [1700, 320, 4.8, 1900, 150, 1400, 50, 1200, 1100, 28.0, 65.0, 18.0],
    'Chhattisgarh': [1600, 280, 4.2, 1750, 130, 1250, 40, 1150, 1050, 26.0, 58.0, 17.0],
    'Goa': [1300, 90, 1.8, 1100, 50, 900, 18, 850, 800, 22.0, 45.0, 12.0],
    'Gujarat': [1800, 340, 5.0, 2000, 170, 1500, 55, 1350, 1200, 31.0, 62.0, 22.0],
    'Haryana': [1750, 330, 4.9, 1950, 160, 1450, 52, 1300, 1150, 30.0, 63.0, 21.0],
    'Himachal Pradesh': [1200, 80, 1.5, 1000, 45, 800, 15, 750, 700, 10.0, 48.0, 8.0],
    'Jharkhand': [1650, 310, 4.4, 1850, 145, 1350, 48, 1250, 1100, 27.0, 64.0, 19.0],
    'Karnataka': [1500, 220, 3.2, 1600, 100, 1150, 30, 1050, 950, 24.0, 55.0, 13.0],
    'Kerala': [1400, 190, 2.8, 1500, 85, 1100, 28, 950, 900, 21.0, 52.0, 12.0],
    'Madhya Pradesh': [1700, 340, 4.8, 1900, 150, 1400, 50, 1300, 1200, 29.0, 65.0, 20.0],
    'Maharashtra': [1800, 350, 5.2, 2000, 170, 1500, 55, 1350, 1250, 30.0, 63.0, 22.0],
    'Manipur': [1300, 90, 1.8, 1100, 50, 900, 18, 850, 800, 22.0, 45.0, 12.0],
    'Meghalaya': [1250, 85, 1.6, 1050, 48, 820, 16, 800, 750, 12.0, 44.0, 11.0],
    'Mizoram': [1200, 80, 1.5, 1000, 45, 800, 15, 750, 700, 10.0, 48.0, 8.0],
    'Nagaland': [1150, 75, 1.3, 950, 40, 780, 14, 700, 650, 9.0, 42.0, 7.0],
    'Odisha': [1700, 310, 4.6, 1900, 160, 1400, 52, 1300, 1200, 28.0, 60.0, 19.0],
    'Punjab': [1800, 340, 5.0, 2000, 170, 1500, 55, 1350, 1250, 30.0, 63.0, 22.0],
    'Rajasthan': [1750, 320, 4.7, 1950, 160, 1450, 50, 1300, 1200, 29.0, 64.0, 21.0],
    'Sikkim': [1100, 70, 1.2, 900, 38, 720, 13, 680, 600, 8.0, 40.0, 7.0],
    'Tamil Nadu': [1500, 220, 3.2, 1600, 100, 1150, 30, 1050, 950, 24.0, 55.0, 13.0],
    'Telangana': [1600, 280, 4.2, 1750, 130, 1250, 40, 1150, 1050, 26.0, 58.0, 17.0],
    'Tripura': [1400, 200, 2.8, 1300, 90, 1100, 25, 950, 900, 21.0, 50.0, 14.0],
    'Uttar Pradesh': [1850, 360, 5.5, 2100, 180, 1550, 60, 1400, 1300, 32.0, 65.0, 23.0],
    'Uttarakhand': [1250, 100, 1.7, 1150, 60, 900, 20, 850, 800, 18.0, 50.0, 10.0],
    'West Bengal': [1750, 320, 4.7, 1950, 150, 1450, 52, 1300, 1200, 28.0, 62.0, 20.0],
}

# Streamlit App UI
def main():
    st.title('Air Quality Prediction App for Indian States')

    st.write("This app predicts the air quality index based on predefined sensor data for each Indian state.")

    # Load and train model
    X, y = load_data()
    model, mse = train_model(X, y)
    st.write(f'Model trained with Mean Squared Error: {mse:.2f}')

    # State selection dropdown
    state = st.selectbox('Select a State in India', list(state_sensor_data.keys()))

    # Button for making prediction
    if st.button('Predict Air Quality Index'):
        input_data = state_sensor_data[state]
        prediction = predict(model, input_data)
        st.success(f'The predicted Air Quality Index (CO concentration) for {state} is: {prediction:.2f}')

if __name__ == "__main__":
    main()
