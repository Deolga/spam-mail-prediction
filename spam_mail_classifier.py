import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "mail_data.csv"
raw_mail_data = pd.read_csv(DATA_PATH)
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), "")
print("Dataset shape:", mail_data.shape)
print(mail_data["Category"].value_counts())

# Encode spam=0, ham=1 to match the supplied project
mail_data["Category"] = mail_data["Category"].map({"spam":0,"ham":1}).astype(int)
X = mail_data["Message"]
Y = mail_data["Category"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3, stratify=Y
)

vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)
print("Training TF-IDF shape:", X_train_features.shape)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_features, y_train)

train_pred = model.predict(X_train_features)
test_pred = model.predict(X_test_features)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print("Confusion matrix [rows=true, cols=predicted]:")
print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, target_names=["spam", "ham"]))

examples = [
    "Congratulations! You have won a free prize. Click now to claim your reward.",
    "Hi, are we still meeting for class tomorrow at 10?"
]
example_pred = model.predict(vectorizer.transform(examples))
for msg, pred in zip(examples, example_pred):
    print("SPAM" if pred == 0 else "HAM", "|", msg)
