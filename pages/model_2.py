import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_model_bundle():
    return {
        "encoders": joblib.load(BASE_DIR / "encoders" / "encoders_car_dataset.joblib"),
        "scaler": joblib.load(BASE_DIR / "scalers" / "scaler_car_dataset.joblib"),
        "model": joblib.load(BASE_DIR / "models" / "rf_model_car_dataset.joblib"),
        "uniques": joblib.load(BASE_DIR / "uniques" / "uniques_car_dataset.joblib"),
    }


def _prepare_features(kms_driven, present_price, fuel_type, seller_type, transmission, age, encoders, scaler, uniques):
    encoded = []
    for value, encoder, options in zip(
        [fuel_type, seller_type, transmission],
        encoders,
        uniques,
    ):
        values_list = [str(v) for v in options]
        code = str(value) if str(value) in values_list else value
        if str(code) in values_list:
            code = values_list.index(str(code))
        encoded.append(float(encoder.transform([code])[0]))

    values = np.array(
        [float(kms_driven), float(present_price), encoded[0], encoded[1], encoded[2], float(age)],
        dtype=float,
    ).reshape(1, -1)

    return scaler.transform(values)


def predict_single(kms_driven, present_price, fuel_type, seller_type, transmission, age):
    bundle = load_model_bundle()
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    uniques = bundle["uniques"]

    X = _prepare_features(
        kms_driven, present_price, fuel_type, seller_type, transmission, age, encoders, scaler, uniques
    )
    pred = model.predict(X)[0]
    return float(pred)


def predict_dataframe(df):
    bundle = load_model_bundle()
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    uniques = bundle["uniques"]

    required_columns = ["Kms_Driven", "Present_Price", "Fuel_Type", "Seller_Type", "Transmission", "Age"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    results = []
    for _, row in df[required_columns].iterrows():
        X = _prepare_features(
            row["Kms_Driven"],
            row["Present_Price"],
            row["Fuel_Type"],
            row["Seller_Type"],
            row["Transmission"],
            row["Age"],
            encoders,
            scaler,
            uniques,
        )
        results.append(float(model.predict(X)[0]))

    df = df.copy()
    df["prediction"] = results
    return df


st.title("Modèle 2 - Car Dataset")
st.caption("Prédiction du prix de vente d'une voiture à partir du dataset Car Dataset.")

tab_simple, tab_multiple = st.tabs(["Prédiction simple", "Prédiction multiple"])

with tab_simple:
    bundle = load_model_bundle()
    uniques = bundle["uniques"]

    with st.form("car_dataset_single_form"):
        col1, col2 = st.columns(2)

        with col1:
            kms_driven = st.number_input("Kms_Driven", min_value=0.0, value=20000.0, step=500.0)
            present_price = st.number_input("Present_Price", min_value=0.0, value=5.0, step=0.5)
            fuel_type = st.selectbox("Fuel_Type", options=uniques[0])

        with col2:
            seller_type = st.selectbox("Seller_Type", options=uniques[1])
            transmission = st.selectbox("Transmission", options=uniques[2])
            age = st.number_input("Age", min_value=0, max_value=30, value=5, step=1)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        prediction = predict_single(kms_driven, present_price, fuel_type, seller_type, transmission, age)
        st.success(f"Prix estimé : {prediction:.2f}")

with tab_multiple:
    st.subheader("Prédiction sur plusieurs voitures")
    st.write("Téléversez un CSV contenant : Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age.")

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
                    file_name="car_dataset_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Erreur lors de la prédiction : {exc}")
