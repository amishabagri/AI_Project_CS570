import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import time
from liveness import liveliness
# Constants
size = 4
datasets = 'datasets'
IMG_SIZE = 100

# Function to create CNN model

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess images and labels
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            images.append(image)
            labels.append(label)
        id += 1

# Convert lists to numpy arrays
images = np.array(images) / 255.0  # Normalize pixel values
labels = to_categorical(labels, len(names))  # Adjusted for num_classes

# Create CNN model
model = create_model((IMG_SIZE, IMG_SIZE, 1), len(names))  # Adjusted for num_classes

# Train the model
model.fit(images.reshape(-1, IMG_SIZE, IMG_SIZE, 1), labels, epochs=10, batch_size=20)
model.save("face_recognition_model.h5")
# Function to predict faces using the trained CNN model
def predict_face(face):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face_normalized = np.array(face_resized) / 255.0
    face_final = np.expand_dims(face_normalized, axis=-1)  # Add an extra dimension for the CNN model
    prediction = model.predict(np.array([face_final]))[0]  # Predictions return a list, we take the first element
    predicted_id = np.argmax(prediction)
    confidence = prediction[predicted_id]
    return predicted_id, confidence

# Face detection and recognition loop
# Face detection and recognition loop
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)
cnt = 0
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    live_detect = liveliness()
    is_face_detected = live_detect.detect_face(im)
    is_live = live_detect.depth_liveness_detection(im)

    if is_face_detected and is_live:
        print("Live face detected")
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face = im[y:y + h, x:x + w]  # Use colored face
            prediction_id, confidence = predict_face(face)  # Pass colored face
            if confidence > 0.8:
                if prediction_id in names:
                    cv2.putText(im, f'{names[prediction_id]} - {confidence:.2f}', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
                    print("identified person: ",names[prediction_id])
                else:
                    cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                cnt = 0
                cv2.destroyAllWindows()  # Close the OpenCV window
                exit()
            else:
                # cnt += 1
                cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                print("Unknown Person")
                # if cnt > 100:
                #     print("Unknown Person")
                #     cv2.imwrite("input.jpg", im)
                #     cnt = 0
                exit()
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    else:
        print("Not a live face")

webcam.release()
cv2.destroyAllWindows()
