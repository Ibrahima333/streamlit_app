import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_model_bundle():
    return {
        "encoders": joblib.load(BASE_DIR / "encoders" / "encoders_taxi.joblib"),
        "scaler": joblib.load(BASE_DIR / "scalers" / "scaler_taxi.joblib"),
        "model": joblib.load(BASE_DIR / "models" / "rf_model_taxi.joblib"),
        "uniques": joblib.load(BASE_DIR / "uniques" / "uniques_taxi.joblib"),
    }


def predict_single(trip_distance, time_of_day, day_of_week, passenger_count, traffic, weather, base_fare, per_km, per_min, duration):
    bundle = load_model_bundle()
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    encoded = []
    for value, encoder in zip([time_of_day, day_of_week, traffic, weather], encoders):
        encoded.append(float(encoder.transform([str(value)])[0]))

    values = np.array(
        [trip_distance, encoded[0], encoded[1], passenger_count, encoded[2], encoded[3], base_fare, per_km, per_min, duration],
        dtype=float,
    ).reshape(1, -1)
    values = scaler.transform(values)
    pred = model.predict(values)[0]
    return float(pred)


def predict_dataframe(df):
    bundle = load_model_bundle()
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    required_columns = [
        "Trip_Distance_km",
        "Time_of_Day",
        "Day_of_Week",
        "Passenger_Count",
        "Traffic_Conditions",
        "Weather",
        "Base_Fare",
        "Per_Km_Rate",
        "Per_Minute_Rate",
        "Trip_Duration_Minutes",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    results = []
    for _, row in df[required_columns].iterrows():
        encoded = []
        for value, encoder in zip([row["Time_of_Day"], row["Day_of_Week"], row["Traffic_Conditions"], row["Weather"]], encoders):
            encoded.append(float(encoder.transform([str(value)])[0]))

        feature_row = np.array(
            [
                row["Trip_Distance_km"],
                encoded[0],
                encoded[1],
                row["Passenger_Count"],
                encoded[2],
                encoded[3],
                row["Base_Fare"],
                row["Per_Km_Rate"],
                row["Per_Minute_Rate"],
                row["Trip_Duration_Minutes"],
            ],
            dtype=float,
        ).reshape(1, -1)
        feature_row = scaler.transform(feature_row)
        results.append(float(model.predict(feature_row)[0]))

    df = df.copy()
    df["prediction"] = results
    return df


st.title("Modèle 3 - Taxi")
st.caption("Prédiction du prix d'un trajet de taxi à partir des caractéristiques du trajet.")

tab_simple, tab_multiple = st.tabs(["Prédiction simple", "Prédiction multiple"])

with tab_simple:
    bundle = load_model_bundle()
    uniques = bundle["uniques"]

    with st.form("taxi_single_form"):
        col1, col2 = st.columns(2)

        with col1:
            trip_distance = st.number_input("Trip_Distance_km", min_value=0.0, value=5.0, step=0.5)
            time_of_day = st.selectbox("Time_of_Day", options=uniques[0])
            day_of_week = st.selectbox("Day_of_Week", options=uniques[1])
            passenger_count = st.number_input("Passenger_Count", min_value=1, max_value=8, value=2, step=1)

        with col2:
            traffic = st.selectbox("Traffic_Conditions", options=uniques[2])
            weather = st.selectbox("Weather", options=uniques[3])
            base_fare = st.number_input("Base_Fare", min_value=0.0, value=3.5, step=0.1)
            per_km = st.number_input("Per_Km_Rate", min_value=0.0, value=1.2, step=0.05)
            per_min = st.number_input("Per_Minute_Rate", min_value=0.0, value=0.3, step=0.01)
            duration = st.number_input("Trip_Duration_Minutes", min_value=0, value=20, step=1)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        prediction = predict_single(
            trip_distance,
            time_of_day,
            day_of_week,
            passenger_count,
            traffic,
            weather,
            base_fare,
            per_km,
            per_min,
            duration,
        )
        st.success(f"Prix estimé : {prediction:.2f}")

with tab_multiple:
    st.subheader("Prédiction sur plusieurs trajets")
    st.write("Téléversez un CSV contenant : Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions, Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes.")

    uploaded_file = st.file_uploader("Téléverser un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Lancer la prédiction"):
            try:
                result = predict_dataframe(df)
                st.dataframe(result)
                st.download_button(
                    "Télécharger le CSV prédit",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="taxi_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Erreur lors de la prédiction : {exc}")
