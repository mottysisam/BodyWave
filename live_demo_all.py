import cv2
import mediapipe as mp
import SquatPosture as sp
import BodyWave as bw  # Importing the BodyWave module
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import *
from csv import writer

csv_file = open('plotting_live.csv', 'w+')
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
writer_object = writer(csv_file)
writer_object.writerow(['neck', 'knee', 'hip', 'ankle', 'y-knee'])

# For video input:
cap = cv2.VideoCapture(0)

# Load models for both squats and bodywave
squat_model = tf.keras.models.load_model("squat_model")
bodywave_model = tf.keras.models.load_model("bodywave_model")  # Assuming you have a separate model for bodywave

counter_for_renewal = 0
with mp_pose.Pose() as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract posture parameters for both squats and bodywave
        squat_params = sp.get_params(results, all=True)
        bodywave_params = bw.get_params(results, all=True)

        # Predict posture using the appropriate model
        squat_output = squat_model.predict(squat_params.T)
        bodywave_output = bodywave_model.predict(bodywave_params.T)

        # Logic to determine which exercise is being performed and label accordingly
        # This is a basic example, and you might need more sophisticated logic based on your model's outputs
        if np.argmax(squat_output) > np.argmax(bodywave_output):
            label = "Squat"
        else:
            label = "BodyWave"

        label_final_results(image, label)

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

csv_file.close()
cap.release()
cv2.destroyAllWindows()
