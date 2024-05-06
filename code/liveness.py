import os
import sys

import cv2
import numpy as np
import dlib

# this is maximum attempts to detect the spoof image
MAX_ATTEMPTS = 2
attempt = 0

# Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
p = os.path.dirname(ROOT_DIR)
shape_predictor_path = os.path.join('shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(shape_predictor_path)

# Initialize Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class liveliness:
    def depth_liveness_detection(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Perform face detection using Haar cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # If no faces detected, return False
        if len(faces) == 0:
            return False

        # Loop over detected faces
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) containing the face
            face_roi = gray[y:y + h, x:x + w]
            cv2.rectangle(face_roi, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
            # Calculate depth variation within the face region (simulated)
            depth_variation = np.std(face_roi)

            # Set a threshold for depth variation to detect spoof images
            depth_variation_threshold = 15

            # Perform liveness detection based on depth analysis
            if depth_variation > depth_variation_threshold:
                # depth detected
                return True

        # If no depth detected
        return False

    # Function to detect eye blinking
    def detect_blink(self, frame, landmarks):
        # Extract eye landmarks from Dlib shape predictor result
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Compute eye aspect ratio for left and right eyes
        left_ear = self.eye_aspect_ratio(left_eye_pts)
        right_ear = self.eye_aspect_ratio(right_eye_pts)

        # Average eye aspect ratio
        avg_ear = (left_ear + right_ear) / 2.0

        # Set a threshold for detecting blinking
        blink_threshold = 0.2
        print("avg ear", avg_ear)
        # checking if the avg_ear is less than threshold returns true
        if avg_ear < blink_threshold:
            return True
        else:
            return False

    # Function to compute eye aspect ratio
    def eye_aspect_ratio(self, eye):
        # Compute Euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])

        # Compute the Euclidean distance between the horizontal eye landmark (x, y)-coordinates
        C = np.linalg.norm(eye[0] - eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # Return the eye aspect ratio
        return ear

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces) > 0