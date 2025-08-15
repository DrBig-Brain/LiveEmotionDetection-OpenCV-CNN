import streamlit as st
import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image
try:
    model = tf.keras.models.load_model('cnn.keras')
    emotion = ['Surprised', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    st.set_page_config(page_title="Live Emotion Detection", layout="centered")
    st.title("🎭 Live Emotion Detection")

    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    cap = cv.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("❌ Failed to access webcam")
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv.resize(face, (100, 100)) / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            prediction = model.predict(face_resized, verbose=0)
            emotion_label = np.argmax(prediction)

            cv.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(rgb, emotion[emotion_label], (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        FRAME_WINDOW.image(rgb)

    cap.release()
except:
    st.text("model failed to load, try rerunning the app")
