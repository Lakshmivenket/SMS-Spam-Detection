
import streamlit as st
import pickle

st.set_page_config(
    page_title="ML Spam Detection",
    page_icon="📩"
)

# Load trained Machine Learning model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📩 Machine Learning Spam Detection")
st.write("Enter a message to predict whether it is Spam or Not Spam.")

message = st.text_area("Enter your message:")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_lower = message.lower()

        suspicious_phrases = [
            "account has been blocked",
            "verify your details",
            "click this link",
            "bank account",
            "confirm your account",
            "update your account"
        ]

        prediction = model.predict([message])[0]

        if prediction == "spam" or any(
            phrase in message_lower
            for phrase in suspicious_phrases
        ):
            st.error("🚨 Potential Spam / Suspicious Message")
        else:
            st.success("✅ Likely Not Spam")


    