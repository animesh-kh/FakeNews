import streamlit as st
import requests

st.set_page_config(page_title="Fake News Detector", page_icon="🕵️‍♂️")

st.title("Fake News Detection App")
st.write("Enter a news headline or article and let the AI decide if it’s real or fake!")

# Text input
news_text = st.text_area("Enter the news text here:")

# When user clicks button
if st.button("Predict"):
    if news_text.strip():
        # Make POST request to FastAPI
        url = "http://127.0.0.1:8000/predict"
        response = requests.post(url, json={"text": news_text})

        if response.status_code == 200:
            result = response.json()
            label = result.get("prediction", "Unknown").upper()
            score = result.get("confidence", 0)

            if label == "FAKE":
                st.error(f"🚨 This news seems **FAKE** (confidence: {score:.2f})")
            elif label == "REAL":
                st.success(f"✅ This news seems **REAL** (confidence: {score:.2f})")
            else:
                st.error("Problem")
        else:
            st.warning("Could not connect to backend. Check if FastAPI is running.")
    else:
        st.info("Please enter some text first.")
