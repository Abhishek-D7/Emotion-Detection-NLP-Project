# Emotion Detection NLP Project

Hello, everyone! Today, I'd like to share a fascinating #project I've been working on - an Emotion Detection model using Natural Language Processing ( #NLP ) and Machine Learning.

The project involves training a model to #predict emotions from text data. Here's a brief overview of the process:

1️⃣ Data Preparation : The #datasets, namely “dair-ai/emotion” and “SetFit/emotion”, are sourced from Hugging Face, a popular platform for sharing and using pre-trained models.

2️⃣ Model Building: A pipeline is created using TextVectorization and a neural network model built with TensorFlow. The TextVectorization layer converts the text data into numerical features, making it suitable for input into the neural network. The neural network, consisting of embedding layers, dense layers with ReLU activation, and dropout for regularization, is used for the prediction of emotions. The model leverages SparseCategoricalCrossentropy as the loss function, tailored for multi-class classification where labels are integer-encoded.

3️⃣ Model Training and Evaluation: The model is #trained on the train shape: (36000, 3) and #predictions are made on the test shape: (4000, 3). The accuracy of the model Accuracy: 0.99794. Additionally, a classification report and confusion matrix are generated for a more detailed performance analysis.

4️⃣ Deployment: The trained model is #deployed using Streamlit, a popular open-source app #framework for Machine Learning and Data Science projects. The app takes user input, predicts the emotion, and displays an image corresponding to the predicted emotion.

🎓 Learning Journey
This project is a great example of how machine learning can be used to understand and interpret human emotions. It has potential applications in areas like customer feedback analysis, social media monitoring, and more.
