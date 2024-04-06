import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

test2 = pd.read_csv(r"D:\Data Sets\NLP\Emotion Detection NLP\test2.csv")
train2 = pd.read_csv(r"D:\Data Sets\NLP\Emotion Detection NLP\train2.csv")
test1 = pd.read_csv(r"D:\Data Sets\NLP\Emotion Detection NLP\test1.csv")
train1 = pd.read_csv(r"D:\Data Sets\NLP\Emotion Detection NLP\train1.csv")
validation2 = pd.read_csv(r"D:\Data Sets\NLP\Emotion Detection NLP\validation2.csv")
validation1 = pd.read_csv(r"D:\Data Sets\NLP\Emotion Detection NLP\validation1.csv")

train3 = pd.concat([train1, train2], ignore_index=True)
test = pd.concat([test1, test2], ignore_index=True)
validation = pd.concat([validation1, validation2], ignore_index=True)
train = pd.concat([train3, validation], ignore_index=True)

print("train shape: ", train.shape)
print("test shape: ", test.shape)

X_train = train['text']
y_train = train['label']
X_test = train['text']
y_test = train['label']

model = Pipeline(steps=[('cv', CountVectorizer()), ('rf', RandomForestClassifier())])
# Train and evaluate model
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
acc_score = accuracy_score(y_test, y_preds)
print("Accuracy: ", acc_score)

# Classification Report
report = classification_report(y_test, y_preds)
print("\nClassification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
print("\nConfusion Matrix:")
print(cm)
