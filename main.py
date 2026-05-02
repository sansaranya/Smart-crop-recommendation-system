import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Input and Output
X = data.drop("label", axis=1)
Y = data["label"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Model creation
model = RandomForestClassifier()

# Train model
model.fit(X_train, Y_train)

# Test prediction
temp_map = {
    "Low": 20,
    "Medium": 28,
    "High": 35
}

humidity_map = {
    "Low": 40,
    "Medium": 65,
    "High": 85
}

rainfall_map = {
    "Low": 50,
    "Medium": 120,
    "High": 200
}

ph_map = {
    "Acidic": 5.5,
    "Normal": 6.5,
    "Basic": 8.0
}

temperature = temp_map["Medium"]
humidity = humidity_map["High"]
rainfall = rainfall_map["High"]
ph = ph_map["Normal"]

N = 90
P = 42
K = 43

prediction = model.predict(pd.DataFrame(
    [[N, P, K, temperature, humidity, ph, rainfall]],
    columns=X.columns
))

print("Recommended Crop is:", prediction[0])