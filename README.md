# Spam Mail Prediction

## Overview
A machine-learning text classifier that predicts whether a message is spam or ham (legitimate). The supplied internship dataset is converted into TF-IDF text features and classified with logistic regression.

## Dataset
`mail_data.csv` is the supplied internship dataset.

## Methodology
1. Load the message dataset.
2. Replace missing values with empty strings.
3. Encode spam as 0 and ham as 1.
4. Split the data into stratified training and test sets.
5. Transform message text into TF-IDF vectors using English stop-word removal.
6. Train a logistic regression classifier.
7. Evaluate training/test accuracy, confusion matrix, and classification report.
8. Demonstrate predictions on example messages.

## Run
```bash
pip install -r requirements.txt
jupyter notebook Spam_Mail_Prediction.ipynb
```

## Limitations
The classifier is trained on the supplied dataset and may not generalize to every language, domain, or newly emerging spam pattern.
