import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_model_bundle():
    return {
        "encoders": joblib.load(BASE_DIR / "encoders" / "encoders_car_data.joblib"),
        "scaler": joblib.load(BASE_DIR / "scalers" / "scaler_car_data.joblib"),
        "model": joblib.load(BASE_DIR / "models" / "dt_model_car_data.joblib"),
        "uniques": joblib.load(BASE_DIR / "uniques" / "uniques_car_data.joblib"),
    }


def predict_single(brand, year, transmission, address, price):
    bundle = load_model_bundle()
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    categorical_values = [brand, transmission, address]
    encoded = [
        float(encoder.transform([str(value)])[0])
        for value, encoder in zip(categorical_values, encoders[:3])
    ]

    features = np.array(
        [encoded[0], float(year), encoded[1], encoded[2], float(price)],
        dtype=float,
    ).reshape(1, -1)

    features = scaler.transform(features)
    pred = model.predict(features)[0]
    return "Venant" if int(pred) == 1 else "Occasion"


def predict_dataframe(df):
    bundle = load_model_bundle()
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    required_columns = ["Marque", "Année", "Boite_Vitesse", "Adresse", "Prix"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df.copy()
    X = df[required_columns].copy()

    categorical_columns = ["Marque", "Boite_Vitesse", "Adresse"]
    for col, encoder in zip(categorical_columns, encoders[:3]):
        X[col] = encoder.transform(X[col].astype(str))

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    df["prediction"] = ["Venant" if int(pred) == 1 else "Occasion" for pred in predictions]
    return df


st.title("Modèle 1 - Car Data")
st.caption("Prédiction de Venant / Occasion à partir des caractéristiques d'une voiture.")

tab_simple, tab_multiple = st.tabs(["Prédiction simple", "Prédiction multiple"])

with tab_simple:
    bundle = load_model_bundle()
    uniques = bundle["uniques"]

    with st.form("car_data_single_form"):
        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox("Marque", options=uniques[0])
            year = st.number_input("Année", min_value=2000, max_value=2035, value=2020, step=1)
            transmission = st.selectbox("Boîte de vitesse", options=uniques[1])

        with col2:
            address = st.selectbox("Adresse", options=uniques[2])
            price = st.number_input("Prix", min_value=0.0, value=15000.0, step=500.0)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        prediction = predict_single(brand, year, transmission, address, price)
        st.success(f"Prédiction : {prediction}")

with tab_multiple:
    st.subheader("Prédiction sur plusieurs voitures")
    st.write("Téléversez un CSV contenant les colonnes : Marque, Année, Boite_Vitesse, Adresse, Prix.")

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
                    file_name="car_data_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
