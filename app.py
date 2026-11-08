import streamlit as st

st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide",
)

pages = [
    st.Page("pages/model_1.py", title="Modèle 1 - Car Data"),
    st.Page("pages/model_2.py", title="Modèle 2 - Car Dataset"),
    st.Page("pages/model_3.py", title="Modèle 3 - Taxi"),
    st.Page("pages/model_4.py", title="Modèle 4 - Telephone"),
]

navigation = st.navigation(pages)
navigation.run()
