import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_model_bundle():
    return {
        "encoders": joblib.load(BASE_DIR / "encoders" / "encoders_telephone.joblib"),
        "scaler": joblib.load(BASE_DIR / "scalers" / "scaler_telephone.joblib"),
        "model": joblib.load(BASE_DIR / "models" / "rf_model_telephone.joblib"),
        "uniques": joblib.load(BASE_DIR / "uniques" / "uniques_telephone.joblib"),
    }


def predict_single(prix,adresse, marque, dim_ecr, ram, stockage):
    bundle = load_model_bundle()
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    encoded = [
        float(encoder.transform([str(value)])[0])
        for value, encoder in zip([adresse, marque], encoders)
    ]

    features = np.array(
        [float(prix) ,encoded[0], encoded[1], float(dim_ecr), float(ram), float(stockage)],
        dtype=float,
    ).reshape(1, -1)

    features = scaler.transform(features)
    pred = model.predict(features)[0]
    classes = {
            0: "Occasion",
            1: "Neuf",
            2: "Reconditionné",
            3: "Venant"
        }
    
    return classes[int(pred)]

def predict_dataframe(df):
    bundle = load_model_bundle()
    encoders = bundle["encoders"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    required_columns = ["prix", "adresse", "marque", "dim_ecr", "ram", "stockage"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df.copy()
    X = df[required_columns].copy()

    categorical_columns = ["adresse", "marque"]
    for col, encoder in zip(categorical_columns, encoders):
        X[col] = encoder.transform(X[col].astype(str))

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    # Correspondance entre les valeurs numériques et les catégories
    classes = {
        0: "Occasion",
        1: "Neuf",
        2: "Reconditionné",
        3: "Venant"
    }

    df["prediction"] = [classes[int(pred)] for pred in predictions]    
    
    return df


st.title("Modèle 4 - Téléphone")
st.caption("Prédiction selon l'adresse, la marque, l'état, la dimension, la RAM et le stockage du téléphone.")

tab_simple, tab_multiple = st.tabs(["Prédiction simple", "Prédiction multiple"])

with tab_simple:
    bundle = load_model_bundle()
    uniques = bundle["uniques"]

    with st.form("telephone_single_form"):
        col1, col2 = st.columns(2)

        with col1:
            adresse = st.selectbox("Adresse", options=uniques[0])
            marque = st.selectbox("Marque", options=uniques[1])

        with col2:
            dim_ecr = st.number_input("dim_ecr", min_value=0.0, value=6.0, step=0.1)
            ram = st.number_input("ram", min_value=0, value=8, step=1)
            stockage = st.number_input("stockage", min_value=0, value=128, step=8)
            prix = st.number_input("Prix", min_value=0.0, value=15000.0, step=500.0)
        submitted = st.form_submit_button("Prédire")

    if submitted:
        prediction = predict_single(prix,adresse, marque, dim_ecr, ram, stockage)
        st.success(f"Prédiction : {prediction}")

with tab_multiple:
    st.subheader("Prédiction sur plusieurs téléphones")
    st.write("Téléversez un CSV contenant : adresse, marque, dim_ecr, ram, stockage.")

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
                    file_name="telephone_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Erreur lors de la prédiction : {exc}")
