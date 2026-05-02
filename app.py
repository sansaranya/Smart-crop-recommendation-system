import streamlit as st
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------- BACKGROUND UI ----------------
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image:
            linear-gradient(rgba(0,0,0,0.80), rgba(0,0,0,0.80)),
            url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        .block-container {
            background: rgba(0, 0, 0, 0.55);
            padding: 2rem;
            border-radius: 15px;
        }

        h1, h2, h3, p, label {
            color: white !important;
        }

        div.stButton > button {
            background-color: #00cc66;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1rem;
            border: none;
            font-weight: bold;
        }

        div.stButton > button:hover {
            background-color: #00aa55;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center; color:#00ff99;'>
🌾 Smart Crop Recommendation System
</h1>
<p style='text-align:center; color:#ddd;'>
AI Powered Agriculture Assistant
</p>
""", unsafe_allow_html=True)

# ---------------- DATA + MODEL ----------------
data = pd.read_csv("Crop_recommendation.csv")

X = data.drop("label", axis=1)
Y = data["label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, Y_train)

# ---------------- WEATHER FUNCTION ----------------
def get_weather():
    lat = 13.0827
    lon = 80.2707

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain"
    res = requests.get(url).json()

    temp = res["current"]["temperature_2m"]
    humidity = res["current"]["relative_humidity_2m"]
    rain = res["current"]["rain"]

    return temp, humidity, rain

# ---------------- INPUT ----------------
st.subheader("Soil Parameters")

col1, col2 = st.columns(2)

with col1:
    N = st.slider("Nitrogen (N)", 0, 200, 90)
    P = st.slider("Phosphorus (P)", 0, 200, 42)

with col2:
    K = st.slider("Potassium (K)", 0, 200, 43)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)

# ---------------- WEATHER MODE ----------------
st.subheader("Weather Input Mode")

mode = st.radio("Choose Mode:", ["Live Weather", "Manual Weather"])

if "temp" not in st.session_state:
    st.session_state.temp = 0
    st.session_state.humidity = 0
    st.session_state.rain = 0

if mode == "Live Weather":
    if st.button("Get Live Weather"):
        temp, humidity, rain = get_weather()

        st.session_state.temp = temp
        st.session_state.humidity = humidity
        st.session_state.rain = rain

        st.success("Live Weather Loaded")

        st.write("Temperature:", temp)
        st.write("Humidity:", humidity)
        st.write("Rainfall:", rain)

else:
    st.session_state.temp = st.slider("Temperature (°C)", 0, 60, 30)
    st.session_state.humidity = st.slider("Humidity (%)", 0, 100, 60)
    st.session_state.rain = st.slider("Rainfall (mm)", 0, 300, 50)

# ---------------- PREDICTION ----------------
if st.button("Predict Crop"):

    temp = st.session_state.temp
    humidity = st.session_state.humidity
    rain = st.session_state.rain

    proba = model.predict_proba([[N, P, K, temp, humidity, ph, rain]])
    top = proba[0].argsort()[-3:][::-1]
    crops = model.classes_

    st.success(f"Best Crop Recommendation: {crops[top[0]]}")

    st.info("Alternative Crops")
    st.write("1:", crops[top[1]])
    st.write("2:", crops[top[2]])