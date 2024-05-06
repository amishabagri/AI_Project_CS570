from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score

# Constants
datasets = 'datasets'
IMG_SIZE = 100

# Load the trained model
model = load_model("face_recognition_model.h5")

# Load names
names = {}
id = 0
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        id += 1

# Load images and labels
images = []
labels = []
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        subject_path = os.path.join(datasets, subdir)
        for filename in os.listdir(subject_path):
            img_path = os.path.join(subject_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(subdir)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoded vectors
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert labels to one-hot encoding
y_train_categorical = np.eye(len(names))[y_train_encoded]
y_test_categorical = np.eye(len(names))[y_test_encoded]

# Reshape images for input to CNN model
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Normalize image data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Compile the model with categorical cross-entropy loss
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Predict labels for test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_categorical, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate F1 score
f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
print(f'F1 Score: {f1:.4f}')

