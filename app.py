import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('xgb_model.pkl')

# Title
st.set_page_config(page_title="Parkinson's Disease Detection", layout="centered")
st.title("🧠 Parkinson's Disease Detection")

st.write("""
This application assists on predicting whether a person has **Parkinson's Disease** based on voice measurements!
""")

# Define all 22 input fields
features = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# App Mode selection
mode = st.radio("Choose input method:", ("📝 Manual Entry", "📂 Upload CSV file"))

if mode == "📝 Manual Entry":
    st.subheader("Enter the following voice measurements:")

    user_inputs = []
    with st.form("manual_form"):
        for feat in features:
            val = st.number_input(feat, format="%.5f")
            user_inputs.append(val)
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_array = np.array(user_inputs).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        if prediction == 1:
            st.error("⚠️ **Parkinson's Disease Detected**.")
        else:
            st.success("✅ **Healthy**.")

elif mode == "📂 Upload CSV":
    st.subheader("Upload a CSV file with **one row per person** (22 columns)")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if list(df.columns) != features:
                st.warning("⚠️ Your CSV must have exactly these 22 columns (in order):")
                st.code(", ".join(features))
            else:
                st.write("📊 Data Preview:")
                st.dataframe(df)

                predictions = model.predict(df)
                df['Prediction'] = ["Parkinson's" if p == 1 else "Healthy" for p in predictions]

                st.success("✅ Predictions complete!")
                st.dataframe(df[['Prediction']])

                # Allow user to download results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", data=csv, file_name="parkinsons_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error reading the file: {e}")