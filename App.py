import streamlit as st
from main import model

emotion_images = {
    "Sadness": r"D:\Data Sets\NLP\Emotion Detection NLP\sadness.jpg",
    "Joy": r"D:\Data Sets\NLP\Emotion Detection NLP\joy.jpg",
    "Love": r"D:\Data Sets\NLP\Emotion Detection NLP\love.png",
    "Anger": r"D:\Data Sets\NLP\Emotion Detection NLP\Anger.jpg",
    "Fear": r"D:\Data Sets\NLP\Emotion Detection NLP\Fear.jpg",
    "Surprise": r"D:\Data Sets\NLP\Emotion Detection NLP\Surprise.jpg"
}

st.title("Emotion Detection App")

with st.form(key="my_form"):
    text = st.text_area("Say something")
    submit_text = st.form_submit_button(label='Submit')
if submit_text:
    predicted_probabilities = model.predict([text])
    predicted_label = np.argmax(predicted_probabilities, axis=1)[0]
    
    label_to_emotion = {
        0: "Sadness",
        1: "Joy",
        2: "Love",
        3: "Anger",
        4: "Fear",
        5: "Surprise"
    }
    
    predicted_emotion = label_to_emotion[predicted_label]
    st.write("Emotion: ", predicted_emotion)

    if predicted_emotion in emotion_images:
        st.image(emotion_images[predicted_emotion], caption=predicted_emotion, use_column_width=True)
    else:
        st.write("Image not found for this emotion")
