import tensorflow as tf
import cv2 as cv
import numpy as np

model = tf.keras.models.load_model('cnn.keras')
emotion = ['Surprized','Fear','Disgust','Happy','Sad','Anger','Neutral']
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.1,minNeighbors=5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h,x:x+w]
        face_resized = cv.resize(face,(100,100))/255.0
        face_resized = np.expand_dims(face_resized,axis=0)

        prediction = model.predict(face_resized)
        emotion_label = np.argmax(prediction)

        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(frame,f"emotion: {emotion[emotion_label]}",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

        cv.imshow("Face detection + emtion classification",frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
print('hello world')