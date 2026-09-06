import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os

# Find the file
print("Files in folder:", os.listdir())

# Load - auto handle any format
if os.path.exists("spam.csv"):
    try:
        df = pd.read_csv("spam.csv", encoding='latin-1')
        print("Original columns:", df.columns.tolist())
        # If columns are not v1,v2, use first 2 columns anyway
        if df.shape[1] >= 3: # some versions have 3 columns
            df = df.iloc[:, :2]
        df.columns = ['label','message']
    except Exception as e:
        print("Trying without header...", e)
        df = pd.read_csv("spam.csv", encoding='latin-1', header=None)
        df = df.iloc[:, :2]
        df.columns = ['label','message']
else:
    df = pd.read_excel("spam.xlsx")
    print("Excel columns:", df.columns.tolist())
    df = df.iloc[:, :2]
    df.columns = ['label','message']

print(df.head())
print(df['label'].value_counts())

# Train
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'].astype(str))
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")

# Final test - MUST be spam
test_msg = ["Congratulations! You won cash prize of $440000 click to know how to claim"]
print("Test result for spam msg:", model.predict(vectorizer.transform(test_msg)))

# Save
with open("model.pkl","wb") as f:
    pickle.dump(model,f)
with open("vectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)

print("SUCCESS! model.pkl and vectorizer.pkl created")