import os
import cv2
import numpy as np
from keras.models import Sequential, load_model  # Add 'load_model' here
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split


# Define your emotions
# emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]

# Function to load and preprocess the dataset
# def load_dataset(data_path):
#     data = []
#     labels = []
#
#     for emotion_label, emotion in enumerate(emotions):
#         emotion_path = os.path.join(data_path, emotion)
#         for img_file in os.listdir(emotion_path):
#             img_path = os.path.join(emotion_path, img_file)
#             img = load_img(img_path, grayscale=True, target_size=(48, 48))
#             img_array = img_to_array(img)
#             data.append(img_array)
#             labels.append(emotion_label)
#
#     data = np.array(data, dtype="float") / 255.0
#     labels = to_categorical(labels, num_classes=len(emotions))
#
#     return data, labels
#
# # Function to build the CNN model
# def build_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(len(emotions), activation='softmax'))
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
#
# # Train the model
# data_path = '/var/www/python/archive/train'
# data, labels = load_dataset(data_path)
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
#
# model = build_model()
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
# model.save('emotion_model.keras')

# Test the model on live video from the laptop camera
model = load_model('emotion_model.keras')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)

        emotion_preds = model.predict(face_roi)[0]
        emotion_label = emotions[np.argmax(emotion_preds)]
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
