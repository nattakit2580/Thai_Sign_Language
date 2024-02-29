import cv2
import numpy as np
import pandas as pd
import pickle
import os
import mediapipe as mp
import time

# Import necessary modules and libraries
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize hand landmarks variables
left_hand_landmarks = None
right_hand_landmarks = None

# Load the pre-trained model for sign language prediction
with open('Thai_Sign_Language.pkl', 'rb') as f:
    model = pickle.load(f)

# Specify the image file path for prediction
image_file_path = 'captured_image.png'

# Read the image
image = cv2.imread(image_file_path)
if image is None:
    print(f"Error: Unable to read the image at {image_file_path}")
    exit()

# Image processing loop
start_time = time.time()
while True:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Pose detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results_pose = pose.process(image_rgb)

    # Hand landmarks detection
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        results_hands = hands.process(image_rgb)

    image_rgb.flags.writeable = True
    image_rgb_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw landmarks for Pose
    mp_drawing.draw_landmarks(image_rgb_bgr, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    # Draw landmarks for Hands
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            if hand_landmarks == results_hands.multi_hand_landmarks[0]:
                left_hand_landmarks = hand_landmarks
                mp_drawing.draw_landmarks(image_rgb_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
            elif hand_landmarks == results_hands.multi_hand_landmarks[1]:
                right_hand_landmarks = hand_landmarks
                mp_drawing.draw_landmarks(image_rgb_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

    try:
        # Extract landmarks for Pose
        pose_landmarks = results_pose.pose_landmarks
        if pose_landmarks:
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks.landmark]).flatten())
        else:
            pose_row = []

        # Extract landmarks for Left Hand
        left_hand_landmarks = results_hands.multi_hand_landmarks[0] if results_hands.multi_hand_landmarks else None
        if left_hand_landmarks:
            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand_landmarks.landmark]).flatten())
        else:
            left_hand_row = []

        # Extract landmarks for Right Hand
        right_hand_landmarks = results_hands.multi_hand_landmarks[1] if len(results_hands.multi_hand_landmarks) > 1 else None
        if right_hand_landmarks:
            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand_landmarks.landmark]).flatten())
        else:
            right_hand_row = []

        # Combine all landmarks into one row
        row = left_hand_row + right_hand_row

        # Predict sign language using the loaded model
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]

        # Display prediction results on the image
        coords = tuple(np.multiply(
            np.array(
                (results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x, 
                 results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y))
            , [image.shape[1], image.shape[0]]).astype(int))
        
        cv2.rectangle(image_rgb_bgr, 
                      (coords[0], coords[1] + 5), 
                      (coords[0] + len(body_language_class) * 20, coords[1] - 30), 
                      (245, 117, 16), -1)
        cv2.putText(image_rgb_bgr, body_language_class, coords, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display status box
        cv2.rectangle(image_rgb_bgr, (0, 0), (250, 60), (245, 117, 16), -1)
        
        # Display class name
        cv2.putText(image_rgb_bgr, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image_rgb_bgr, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display probability
        cv2.putText(image_rgb_bgr, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image_rgb_bgr, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Save the predicted image
        cv2.imwrite('predicted_image.png', image_rgb_bgr)

        # Display the image with landmarks and predictions
        cv2.imshow('Image with Landmarks and Predictions', image_rgb_bgr)

        # Check if 3 seconds have passed
        if time.time() - start_time > 3:
            break

    except:
        pass

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
