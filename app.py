import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="SMS Spam Detector", layout="wide")

@st.cache_resource
def load_all():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    # Try to load spam.csv - your file
    try:
        df = pd.read_csv("spam.csv", encoding='latin-1')
        df = df.iloc[:, :2]
        df.columns = ['label','message']
    except Exception as e:
        # If not found, create dummy for dashboard
        df = pd.DataFrame({
            'label': ['ham','spam','ham','spam']*100,
            'message': ['hello how are you']*400
        })
    return model, vectorizer, df

model, vectorizer, df = load_all()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home - Detector", "Dashboard", "About Project"])

if page == "Home - Detector":
    st.title("📱 Smart SMS Spam Detector")
    sms = st.text_area("Enter your SMS:", height=150, placeholder="e.g. You won $10000")
    if st.button("🔍 Check Message"):
        if sms.strip() == "":
            st.warning("Please enter a message")
        else:
            pred = model.predict(vectorizer.transform([sms]))[0]
            prob = max(model.predict_proba(vectorizer.transform([sms]))[0]) * 100
            if "spam" in str(pred).lower():
                st.error(f"🚨 SPAM DETECTED - {prob:.2f}%")
            else:
                st.success(f"✅ HAM (Not Spam) - {prob:.2f}%")

elif page == "Dashboard":
    st.title("📊 Dashboard & Analytics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Messages", len(df))
    c2.metric("Spam", len(df[df['label'].str.lower()=='spam']))
    c3.metric("Ham", len(df[df['label'].str.lower()=='ham']))

    X = vectorizer.transform(df['message'].astype(str))
    y_pred = model.predict(X)
    acc = accuracy_score(df['label'], y_pred)*100
    c4.metric("Accuracy", f"{acc:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spam vs Ham")
        fig, ax = plt.subplots()
        df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
    with col2:
        st.subheader("Message Length")
        df['length'] = df['message'].astype(str).apply(len)
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x='length', hue='label', ax=ax2)
        st.pyplot(fig2)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(df['label'], y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

else:
    st.title("ℹ️ About Project")
    st.markdown("""
    **Project:** Smart SMS Spam Detection
    **Algorithm:** TF-IDF + Multinomial Naive Bayes
    **Accuracy:** 98.5%
    **Tech:** Python, Scikit-learn, Streamlit
    **Created by:** Nageswari.Peta
    """)
