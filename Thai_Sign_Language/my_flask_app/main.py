import mediapipe as mp
import cv2
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy
import os

# นำเข้าโมดูลและไลบรารีที่ต้องใช้
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# สร้างตัวแปรเพื่อเก็บ landmark ของมือซ้ายและมือขวา
left_hand_landmarks = None
right_hand_landmarks = None

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# โหลดโมเดลที่ถูกบันทึกไว้ในไฟล์และเตรียมโมเดลสำหรับการทำนายภาษากาย
with open('Thai_Sign_Language_Term2.pkl', 'rb') as f:
    model = pickle.load(f)

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # แปลงภาพเฟรมเป็นรูปแบบสี RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # # ทำการตรวจจับพิกัด (landmarks) สำหรับการตรวจจับท่าทาง (Pose)
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #     results_pose = pose.process(image)

    # ทำการตรวจจับพิกัด (landmarks) สำหรับมือ (Hands)
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        results_hands = hands.process(image)

    # ทำการตรวจจับใบหน้า (Face Detection)
    #with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        #results_face_detection = face_detection.process(image)

    # ทำการตรวจจับพิกัด (landmarks) สำหรับใบหน้า (Face Mesh)
    #with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        #results_face_mesh = face_mesh.process(image)

    # แปลงภาพกลับเป็นรูปแบบ BGR เพื่อแสดงผล
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # # วาดพิกัด (landmarks) สำหรับท่าทาง (Pose)
    # mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
    #                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    # วาดพิกัด (landmarks) สำหรับมือ (Hands)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            if hand_landmarks == results_hands.multi_hand_landmarks[0]:
                left_hand_landmarks = hand_landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
            elif hand_landmarks == results_hands.multi_hand_landmarks[1]:
                right_hand_landmarks = hand_landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

    # วาดผลลัพธ์การตรวจจับใบหน้า (Face Detection)
    #if results_face_detection.detections:
        #for detection in results_face_detection.detections:
            #mp_drawing.draw_detection(image, detection)

    # วาดพิกัด (landmarks) สำหรับใบหน้า (Face Mesh)
    #if results_face_mesh.multi_face_landmarks:
        #for face_landmarks in results_face_mesh.multi_face_landmarks:
            #mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                     #mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                     #mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    # แสดงภาพ
    cv2.imshow('Webcam Feed with Landmarks', image)

    # ทำการทำนายภาษากายโดยใช้โมเดลที่โหลดมา
    try:
        # แยกพิกัดสำหรับท่าทาง (Pose landmarks)
        pose_landmarks = results_pose.pose_landmarks
        if pose_landmarks:
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks.landmark]).flatten())
        else:
            pose_row = []

        # # แยกพิกัดสำหรับใบหน้า (Face landmarks)
        # face_landmarks = results_face_mesh.multi_face_landmarks[0] if results_face_mesh.multi_face_landmarks else None
        # if face_landmarks:
        #     face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face_landmarks.landmark]).flatten())
        # else:
        #     face_row = []

        # แยกพิกัดสำหรับมือซ้าย (Left Hand landmarks)
        left_hand_landmarks = results_hands.multi_hand_landmarks[0] if results_hands.multi_hand_landmarks else None
        if left_hand_landmarks:
            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand_landmarks.landmark]).flatten())
        else:
            left_hand_row = []

        # แยกพิกัดสำหรับมือขวา (Right Hand landmarks)
        right_hand_landmarks = results_hands.multi_hand_landmarks[1] if len(results_hands.multi_hand_landmarks) > 1 else None
        if right_hand_landmarks:
            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand_landmarks.landmark]).flatten())
        else:
            right_hand_row = []

        # รวมพิกัดทั้งหมดเข้าด้วยกัน
        row =  left_hand_row + right_hand_row # + face_row  pose_row pose_row +

        # ทำการทำนายภาษากายโดยใช้โมเดลที่โหลดมา
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]

        # แสดงผลลัพธ์การทำนาย
        coords = tuple(np.multiply(
            np.array(
                (results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x, 
                 results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y))
            , [640, 480]).astype(int))
        
        cv2.rectangle(image, 
                      (coords[0], coords[1] + 5), 
                      (coords[0] + len(body_language_class) * 20, coords[1] - 30), 
                      (245, 117, 16), -1)
        cv2.putText(image, body_language_class, coords, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # แสดงกล่องสถานะ
        cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
        
        # แสดงชื่อคลาส
        cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # แสดงความน่าจะเป็น
        cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    except:
        pass

    # แสดงภาพกลับพร้อมพิกัดและการทำนาย
    cv2.imshow('Webcam Feed with Landmarks', image)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()