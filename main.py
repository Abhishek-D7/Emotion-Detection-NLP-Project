import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
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

# Parameters
max_features = 10000
sequence_length = 100

# Text Vectorization Layer
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Adapt the layer to the text data
vectorize_layer.adapt(X_train.values)

# Define the Neural Network Model
model = Sequential([
    vectorize_layer,
    Embedding(max_features + 1, 128),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(y_train.unique()), activation='softmax')  # Multi-class classification
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Predict and evaluate
y_preds = model.predict(X_test)
y_preds = y_preds.argmax(axis=1)  # Get the class with the highest probability
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
