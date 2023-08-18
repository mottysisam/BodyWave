import cv2
import mediapipe as mp
import math
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

EXERCISES = [
    'bodywave'
]

def get_angle(v1, v2):
    dot = np.dot(v1, v2)
    mod_v1 = np.linalg.norm(v1)
    mod_v2 = np.linalg.norm(v2)
    cos_theta = dot/(mod_v1*mod_v2)
    theta = math.acos(cos_theta)
    return theta

def get_length(v):
    return np.dot(v, v)**0.5

def get_params(results, exercise='bodywave', all=False):

    if results.pose_landmarks is None:
        return np.array([0, 0, 0, 0, 0])

    points = {}
    # Extracting all the necessary landmarks from the results
    for landmark in mp_pose.PoseLandmark:
        points[landmark.name] = np.array([results.pose_landmarks.landmark[landmark].x,
                                          results.pose_landmarks.landmark[landmark].y,
                                          results.pose_landmarks.landmark[landmark].z])

    points["MID_SHOULDER"] = (points["LEFT_SHOULDER"] + points["RIGHT_SHOULDER"]) / 2
    points["MID_HIP"] = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2

    # ... [same as in the provided code for calculating angles and lengths]

    if exercise == 'bodywave':
        # Angle between chest and hips
        theta_chest_hips = get_angle(points["MID_SHOULDER"] - points["MID_HIP"], np.array([0, 0, 1]))
    
        # Spinal curve - using neck, mid-spine (between shoulders and hips), and hips
        theta_neck_spine = get_angle(points["NOSE"] - points["MID_SHOULDER"], points["MID_HIP"] - points["MID_SHOULDER"])
        theta_spine_hips = get_angle(points["MID_SHOULDER"] - points["MID_HIP"], np.array([0, 0, 1]))
    
        # Hip movement - forward and backward tilt
        theta_hip_tilt = get_angle(points["LEFT_HIP"] - points["RIGHT_HIP"], np.array([0, 1, 0]))
    
        # Distance between chest and hips
        distance_chest_hips = get_length(points["MID_SHOULDER"] - points["MID_HIP"])
    
        params = np.array([theta_chest_hips, theta_neck_spine, theta_spine_hips, theta_hip_tilt, distance_chest_hips])

    if all:
        params = np.array([[x, y, z] for pos, (x, y, z) in points.items()])

    return np.round(params, 2)
