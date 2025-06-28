import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
data = pd.read_csv("traffic volume.csv")

# Fill missing values
data.fillna({
    'temp': data['temp'].mean(),
    'rain': data['rain'].mean(),
    'snow': data['snow'].mean()
}, inplace=True)
data['weather'] = data['weather'].fillna('Clouds')

# Create datetime features
data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['Time'], dayfirst=True)
data['day'] = data['datetime'].dt.day
data['month'] = data['datetime'].dt.month
data['year'] = data['datetime'].dt.year
data['hours'] = data['datetime'].dt.hour
data['minutes'] = data['datetime'].dt.minute
data['seconds'] = data['datetime'].dt.second

# Label encode categorical columns
for col in ['holiday', 'weather']:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Features and target
features = ['holiday', 'temp', 'rain', 'snow', 'weather',
            'year', 'month', 'day', 'hours', 'minutes', 'seconds']
X = data[features]
y = data['traffic_volume']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scale.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ model.pkl and scale.pkl created successfully.")
