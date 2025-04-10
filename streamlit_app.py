import streamlit as st
import torch
from model import LogisticRegressionModel
import joblib

model = LogisticRegressionModel(input_dim=10000)
model.load_state_dict(torch.load('model_1v_500_epochs.pth', map_location=torch.device('cuda')))
model.eval()

vectorizer = joblib.load('vectorizer.pkl')

st.title("📰 Fake News Detector")

input_text = st.text_area("Paste your article below:", height=400)

if st.button("PREDICT"):
    if input_text.strip():

        vectorized_input = vectorizer.transform([input_text]).toarray()
        input_tensor = torch.tensor(vectorized_input, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            
            probabilities = torch.sigmoid(output).item()

            prediction = 1 if probabilities >= 0.5 else 0

            confidence = probabilities if prediction == 1 else (1 - probabilities)

        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

        if confidence >= 0.7:
            bar_color = "green"
        elif confidence >= 0.4:
            bar_color = "yellow"
        else:
            bar_color = "red"

        progress_html = f"""
        <div style="width: 100%; background-color: #e0e0e0; border-radius: 5px;">
            <div style="height: 30px; width: {confidence * 100}%; background-color: {bar_color}; border-radius: 5px;">
                <span style="text-align: center; width: 100%; color: black; font-weight: bold; position: absolute; margin-left: 50%; transform: translateX(-50%);">
                    {int(confidence * 100)}%
                </span>
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        
    else:
        st.warning("Text box empty, please enter text.")