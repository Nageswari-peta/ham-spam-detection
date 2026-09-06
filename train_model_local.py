import pandas as pd

# --- Load data ---
# Update this path to point to wherever your spam.xlsx file actually is
excel_path = "spam.xlsx"

df_excel = pd.read_excel(excel_path)
df_excel.to_csv("spam.csv", index=False)

df = pd.read_csv("spam.csv")
print(df.head())
print(df.shape)
print(df["Category"].value_counts())

# --- Plot class balance ---
counts = df["Category"].value_counts()
counts.plot(kind="bar")
plt.title("Spam v/s Ham")
plt.xlabel("Message type")
plt.ylabel("Number of messages")
plt.tight_layout()
plt.savefig("spam_vs_ham.png")  # saved to file instead of plt.show() so it doesn't block
plt.close()
print("Saved chart to spam_vs_ham.png")

# --- Prepare data ---
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})
x = df["Message"]
y = df["Category"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words="english")
x_train_vector = vectorizer.fit_transform(x_train)
x_test_vector = vectorizer.transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train_vector, y_train)

predictions = model.predict(x_test_vector)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# --- Interactive test ---
message = input("Enter a message: ")
message_vector = vectorizer.transform([message])
prediction = model.predict(message_vector)
print("Spam" if prediction[0] == 1 else "Ham")
