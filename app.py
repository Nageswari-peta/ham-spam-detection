import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="SMS Spam Detector", layout="wide")

# Load model and data
@st.cache_resource
def load_all():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    try:
        df = pd.read_csv("spam.csv", encoding='latin-1')
        df = df.iloc[:, :2]
        df.columns = ['label','message']
    except:
        df = pd.read_excel("spam.xlsx")
        df = df.iloc[:, :2]
        df.columns = ['label','message']
    return model, vectorizer, df

model, vectorizer, df = load_all()

# Sidebar Menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home - Detector", "📊 Dashboard", "ℹ️ About Project"])

if page == "🏠 Home - Detector":
    st.title("📱 Smart SMS Spam Detector")
    st.markdown("### Check any SMS for Spam or Ham")

    sms = st.text_area("Enter your SMS message:", height=150, placeholder="e.g. Congratulations! You won $10000 cash prize...")

    col1, col2 = st.columns([1,4])
    with col1:
        check = st.button("🔍 Check Message", use_container_width=True)
    with col2:
        clear = st.button("Clear", use_container_width=True)

    if check:
        if sms.strip() == "":
            st.warning("Please enter a message")
        else:
            pred = model.predict(vectorizer.transform([sms]))[0]
            prob = max(model.predict_proba(vectorizer.transform([sms]))[0]) * 100

            if "spam" in str(pred).lower():
                st.error(f"🚨 **SPAM DETECTED** - Confidence: {prob:.2f}%")
                st.progress(int(prob))
            else:
                st.success(f"✅ **HAM (Not Spam)** - Confidence: {prob:.2f}%")
                st.progress(int(prob))

elif page == "📊 Dashboard":
    st.title("📊 Project Dashboard & Analytics")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Messages", len(df))
    c2.metric("Spam Count", len(df[df['label'].str.lower()=='spam']))
    c3.metric("Ham Count", len(df[df['label'].str.lower()=='ham']))

    # Accuracy
    X = vectorizer.transform(df['message'].astype(str))
    y_pred = model.predict(X)
    acc = accuracy_score(df['label'], y_pred)*100
    c4.metric("Model Accuracy", f"{acc:.2f}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spam vs Ham Distribution")
        fig, ax = plt.subplots()
        df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff6b6b','#51cf66'], ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)

    with col2:
        st.subheader("Message Length Analysis")
        df['length'] = df['message'].astype(str).apply(len)
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x='length', hue='label', bins=50, ax=ax2)
        st.pyplot(fig2)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(df['label'], y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'], ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

else:
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Project:** Smart SMS Spam Detection System
    **Domain:** Natural Language Processing + Machine Learning
    **Algorithm Used:** TF-IDF Vectorizer + Multinomial Naive Bayes
    **Dataset:** SMS Spam Collection (5574 messages)
    **Tech Stack:** Python, Scikit-learn, Pandas, Streamlit

    **How it works:**
    1. Text is cleaned and converted to numbers using TF-IDF
    2. Naive Bayes model learns spam patterns (like 'win', 'prize', 'urgent', 'free')
    3. New message is predicted as Spam or Ham

    **Created by:** Siv
    **Deployed on:** Streamlit Cloud
    """)
    st.info("Live Link: https://ham-spam-detection-dlnl2lktqb6band74yzq8m.streamlit.app/")
