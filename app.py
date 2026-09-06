import streamlit as st
import joblib

st.set_page_config(
    page_title="Smart SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

st.title("📱 Smart SMS Spam Detector")
st.write("Enter an SMS message below to check whether it is **Spam** or **Ham**.")

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error(
        "Model files are missing. Run `python train_model.py` first "
        "to create model.pkl and vectorizer.pkl."
    )
    st.stop()

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Example: Congratulations! You won a free prize. Call now!",
    height=150
)

if st.button("🔍 Check Message", use_container_width=True):
    if not message.strip():
        st.warning("Please enter an SMS message.")
    else:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
            st.write("This message is predicted to be **Spam**.")
        else:
            st.success("📩 HAM MESSAGE")
            st.write("This message is predicted to be **Ham (not spam)**.")
