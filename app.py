import streamlit as st
import pickle

st.set_page_config(page_title="Smart SMS Spam Detector")
st.title("Smart SMS Spam Detector")
st.write("Enter an SMS to check Spam or Ham")

try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except:
    st.error("Model files missing. Run train_model.py first")
    st.stop()

sms = st.text_area("Enter your SMS message:")

if st.button("Check Message"):
    if sms == "":
        st.warning("Please type a message")
    else:
        result = model.predict(vectorizer.transform([sms]))[0]
        if "spam" in str(result).lower():
            st.error("SPAM MESSAGE - This is Spam")
        else:
            st.success("HAM MESSAGE - This is Not Spam")
