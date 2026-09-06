# Smart SMS Spam Detector

This product converts the Day 5 SMS Spam Detection mini-project into a simple Streamlit web application.

## ML approach from the mini-project

- Dataset: SMS messages with `Category` and `Message` columns
- Labels: `ham` and `spam`
- Ham is mapped to 0
- Spam is mapped to 1
- Train/test split: 80/20
- Text vectorization: `CountVectorizer(stop_words="english")`
- Classification model: `LogisticRegression(max_iter=1000)`

## Project files

```text
SMS_Spam_Detector/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── spam.csv              # add your dataset here
├── model.pkl             # created by train_model.py
└── vectorizer.pkl        # created by train_model.py
```

## Step 1 — Prepare the dataset

Your original Colab notebook first converted an Excel file into `spam.csv`.

If you already have `spam.csv`, put it in this folder.

It must contain:

- `Category` — values such as `ham` and `spam`
- `Message` — the SMS text

If you only have the Excel file, convert it first:

```python
import pandas as pd
df = pd.read_excel("spam.xlsx")
df.to_csv("spam.csv", index=False)
```

## Step 2 — Install Python packages

Open Command Prompt/Terminal in this folder and run:

```bash
pip install -r requirements.txt
```

## Step 3 — Train and save the ML model

Run:

```bash
python train_model.py
```

This creates:

```text
model.pkl
vectorizer.pkl
```

The terminal will also print the model accuracy and confusion matrix.

## Step 4 — Start the product

Run:

```bash
streamlit run app.py
```

A browser window should open with the SMS Spam Detector.

## Step 5 — Test it

Try messages such as:

```text
Congratulations! You have won a free prize. Call now!
```

or:

```text
Hi, I will reach home by 7 PM.
```

The application will display either:

- 🚨 SPAM MESSAGE
- 📩 HAM MESSAGE

## How to explain the product in a project presentation

**Product name:** Smart SMS Spam Detector

**Problem:** Spam SMS messages can be unwanted or potentially harmful. The system automatically classifies an SMS as Spam or Ham.

**Technology:** Python, Pandas, Scikit-learn, CountVectorizer, Logistic Regression, Streamlit.

**Workflow:**

```text
SMS Message
     ↓
CountVectorizer
     ↓
Logistic Regression
     ↓
Spam / Ham Prediction
     ↓
Streamlit Web Interface
```

## Important note

The uploaded Colab notebook contains the ML workflow, but the dataset file itself was not included with the notebook. Therefore, `spam.csv` must be placed in this folder before running `train_model.py`.
