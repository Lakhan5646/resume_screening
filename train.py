'''import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/resume.csv")

# Check columns
print(df.columns)

# Use correct columns
df = df[['Resume_str', 'Category']]

# Rename for simplicity
df.columns = ['Resume', 'Category']

# Drop missing values
df.dropna(inplace=True)

print(df.head())'''


'''import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Download stopwords (run only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ==============================
# 🔹 1. Load Dataset
# ==============================
df = pd.read_csv("data/resume.csv")

# Keep required columns
df = df[['Resume_str', 'Category']]
df.columns = ['Resume', 'Category']

# Remove missing values
df.dropna(inplace=True)

print("Dataset Loaded ✅")
print(df.head())

# ==============================
# 🔹 2. Clean Text
# ==============================
def clean_text(text):
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    text = re.sub(r'\d+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^a-z ]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return " ".join(words)

# Apply cleaning
df['cleaned'] = df['Resume'].apply(clean_text)

df.drop_duplicates(subset='cleaned', inplace=True)

print("\nText Cleaning Done ✅")

counts = df['Category'].value_counts()

df = df.groupby('Category').apply(
    lambda x: x.sample(min(len(x), 200), random_state=42)
).reset_index(drop=True)

# ==============================
# 🔹 3. TF-IDF Vectorization
# ==============================
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['Category']

print("Text Vectorization Done ✅")

# ==============================
# 🔹 4. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 🔹 5. Train Model (SVM)
# ==============================
model = LinearSVC(C=1.0)
model.fit(X_train, y_train)

print("Model Training Done ✅")

# ==============================
# 🔹 6. Evaluate Model
# ==============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# ==============================
# 🔹 7. Save Model
# ==============================
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved ✅")

# ==============================
# 🔹 8. Test Prediction
# ==============================
sample = "Python developer with machine learning and data analysis experience"

cleaned_sample = clean_text(sample)
vec = vectorizer.transform([cleaned_sample])

prediction = model.predict(vec)

print("\nSample Prediction:", prediction[0])

import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ==============================
# 🔹 1. Load Dataset
# ==============================
df = pd.read_csv("data/resume.csv")

# Keep required columns
df = df[['Resume_str', 'Category']]
df.columns = ['Resume', 'Category']

# Remove missing values
df.dropna(inplace=True)

print("Dataset Loaded ✅")
print(df['Category'].value_counts())

# ==============================
# 🔹 2. Clean Text (Improved)
# ==============================
def clean_text(text):
    text = str(text).lower()

    # Remove HTML
    text = re.sub(r'<.*?>', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Remove special characters
    text = re.sub(r'[^a-z ]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    # Remove stopwords + small words
    words = [w for w in words if w not in stop_words and len(w) > 3]

    return " ".join(words)

# Apply cleaning
df['cleaned'] = df['Resume'].apply(clean_text)

# Remove duplicates
df.drop_duplicates(subset='cleaned', inplace=True)

print("\nText Cleaning Done ✅")

# ==============================
# 🔹 3. TF-IDF (Improved)
# ==============================
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['Category']

print("\nText Vectorization Done ✅")

# ==============================
# 🔹 4. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 🔹 5. Train Multiple Models
# ==============================
models = {
    "SVM": LinearSVC(),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Naive Bayes": MultinomialNB()
}

best_model = None
best_accuracy = 0

print("\nModel Training & Comparison 🔥")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {acc:.2f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# ==============================
# 🔹 6. Final Evaluation
# ==============================
y_pred = best_model.predict(X_test)

print(f"\nBest Model Accuracy: {best_accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 🔹 7. Save Best Model
# ==============================
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nBest Model Saved ✅")

# ==============================
# 🔹 8. Test Prediction
# ==============================
sample = "Python developer with machine learning and data analysis experience"

cleaned_sample = clean_text(sample)
vec = vectorizer.transform([cleaned_sample])

prediction = best_model.predict(vec)

print("\nSample Prediction:", prediction[0])'''

import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ==============================
# 🔹 1. LOAD DATA
# ==============================
df = pd.read_csv("data/resume.csv")

# Use required columns
df = df[['Resume_str', 'Category']]
df.columns = ['Resume', 'Category']

# Drop nulls & duplicates
df.dropna(inplace=True)
df.drop_duplicates(subset='Resume', inplace=True)

print("Dataset Loaded ✅")
print(df['Category'].value_counts())
#print(df['Category'].unique())
def simplify_category(cat):
    cat = str(cat).lower()

    if 'engineer' in cat or 'developer' in cat or 'software' in cat:
        return 'TECH'

    elif 'hr' in cat or 'recruit' in cat:
        return 'HR'

    elif 'marketing' in cat or 'digital' in cat or 'media' in cat:
        return 'MARKETING'

    elif 'finance' in cat or 'account' in cat or 'bank' in cat:
        return 'FINANCE'

    else:
        return None   

df['Category'] = df['Category'].apply(simplify_category)
df = df[df['Category'].notnull()]

print(df['Category'].value_counts())
# ==============================
# 🔹 2. CLEAN TEXT (BALANCED CLEANING)
# ==============================
def clean_text(text):
    text = str(text).lower()

    # Remove HTML
    text = re.sub(r'<.*?>', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Remove special characters
    text = re.sub(r'[^a-z ]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    # Keep useful words (not too aggressive)
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)

df['cleaned'] = df['Resume'].apply(clean_text)

print("\nText Cleaning Done ✅")

# ==============================
# 🔹 3. TF-IDF (OPTIMAL SETTINGS)
# ==============================
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['Category']

print("Text Vectorization Done ✅")

# ==============================
# 🔹 4. STRATIFIED SPLIT (CRITICAL)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 🔹 5. FINAL MODEL (BEST CONFIG)
# ==============================
model = LinearSVC(C=1.5, class_weight='balanced')

model.fit(X_train, y_train)

print("Model Training Done ✅")

# ==============================
# 🔹 6. EVALUATION
# ==============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Accuracy: {accuracy:.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=1))

# ==============================
# 🔹 7. SAVE MODEL
# ==============================
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully ✅")

# ==============================
# 🔹 8. TEST PREDICTION
# ==============================
sample = "Python developer with machine learning and data analysis experience"

cleaned_sample = clean_text(sample)
vec = vectorizer.transform([cleaned_sample])

prediction = model.predict(vec)

print("\nSample Prediction:", prediction[0])