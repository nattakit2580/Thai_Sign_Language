# from flask import Flask, render_template, send_from_directory
# from flask_socketio import SocketIO
# from threading import Thread
# import cv2
# import mediapipe as mp
# import pickle
# import os

# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")  # ใช้ * เพื่อรับทุกรา origins หรือใส่ origin ที่ต้องการอนุญาต

# mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# left_hand_landmarks = None
# right_hand_landmarks = None

# model_path = r'C:\Users\asus\Desktop\web detection\my_flask_app\Thai_Sign_Language.pkl'

# with open(model_path, 'rb') as f:
#     model = pickle.load(f)

# video_capture = None

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/index2.html')
# def index2():
#     return render_template('index2.html')

# @app.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# @socketio.on('connect', namespace='/video')
# def test_connect():
#     print('Web client connected')

# @socketio.on('disconnect', namespace='/video')
# def test_disconnect():
#     print('Web client disconnected')

# prediction_started = False

# @socketio.on('start_prediction', namespace='/video')
# def start_prediction():
#     global prediction_started
#     prediction_started = True
#     print('Start prediction request received')

# @socketio.on('stop_prediction', namespace='/video')
# def stop_prediction():
#     global prediction_started
#     prediction_started = False
#     print('Stop prediction request received')

# def start_camera():
#     global video_capture
#     video_capture = cv2.VideoCapture(0)

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
#         mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands, \
#         mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
#         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

#         while True:
#             if video_capture:
#                 success, frame = video_capture.read()
#                 if not success:
#                     break

#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results_pose = pose.process(frame_rgb)
#                 results_hands = hands.process(frame_rgb)
#                 results_face_detection = face_detection.process(frame_rgb)
#                 results_face_mesh = face_mesh.process(frame_rgb)

#                 holistic_data = get_body_language_prediction(frame, results_pose, results_hands, results_face_detection, results_face_mesh)
#                 socketio.emit('holistic_data', holistic_data, namespace='/video')

#                 ret, buffer = cv2.imencode('.webm', frame)
#                 if not ret:
#                     continue

#                 frame_bytes = buffer.tobytes()
#                 socketio.emit('video_stream', frame_bytes, namespace='/video')

# def get_body_language_prediction(frame, results_pose, results_hands, results_face_detection, results_face_mesh):
#     global left_hand_landmarks, right_hand_landmarks

#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False

#     mp.drawing_utils.draw_landmarks(
#         frame, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#             color=(245, 117, 66), thickness=2, circle_radius=4),
#         connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#             color=(245, 66, 230), thickness=2, circle_radius=2))

#     if results_hands.multi_hand_landmarks:
#         for hand_landmarks in results_hands.multi_hand_landmarks:
#             if hand_landmarks == results_hands.multi_hand_landmarks[0]:
#                 left_hand_landmarks = hand_landmarks
#                 mp.solutions.drawing_utils.draw_landmarks(
#                     frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
#                     landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#                         color=(80, 22, 10), thickness=2, circle_radius=4),
#                     connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#                         color=(80, 44, 121), thickness=2, circle_radius=2))
#             elif hand_landmarks == results_hands.multi_hand_landmarks[1]:
#                 right_hand_landmarks = hand_landmarks
#                 mp.solutions.drawing_utils.draw_landmarks(
#                     frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
#                     landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#                         color=(121, 22, 76), thickness=2, circle_radius=4),
#                     connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#                         color=(121, 44, 250), thickness=2, circle_radius=2))

#     if results_face_detection.detections:
#         for detection in results_face_detection.detections:
#             mp.solutions.drawing_utils.draw_detection(
#                 frame, detection)

#     if results_face_mesh.multi_face_landmarks:
#         for face_landmarks in results_face_mesh.multi_face_landmarks:
#             mp.solutions.drawing_utils.draw_landmarks(
#                 frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#                     color=(80, 110, 10), thickness=1, circle_radius=1),
#                 connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#                     color=(80, 256, 121), thickness=1, circle_radius=1))

#     cv2.imshow('Webcam Feed with Landmarks', frame)

#     return {
#         'pose_landmarks': results_pose.pose_landmarks,
#         'left_hand_landmarks': left_hand_landmarks,
#         'right_hand_landmarks': right_hand_landmarks,
#         'face_detection_detections': results_face_detection.detections,
#         'face_mesh_landmarks': results_face_mesh.multi_face_landmarks,
#     }

# if __name__ == '__main__':
#     video_thread = Thread(target=start_camera)
#     video_thread.daemon = True
#     video_thread.start()
#     socketio = SocketIO(app, cors_allowed_origins="*")  # ใช้ * เพื่อรับทุกรา origins หรือใส่ origin ที่ต้องการอนุญาต
#     socketio.run(app, debug=True)




from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Import necessary modules and libraries
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# สร้างตัวแปรเพื่อเก็บ landmark ของมือซ้ายและมือขวา
left_hand_landmarks = None
right_hand_landmarks = None

# Load the pre-trained model for sign language prediction
with open('Thai_Sign_Language_Term2.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to process image, predict sign language, and return results
def process_image(image):
    global left_hand_landmarks, right_hand_landmarks

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
        row = left_hand_row 

        # Predict sign language using the loaded model
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]

        return {
            'body_language_class': body_language_class,
            'body_language_prob': round(body_language_prob[np.argmax(body_language_prob)], 2),
            'image_with_predictions': image_rgb_bgr
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            result = process_image(image)

            if result:
                # Save the image with predictions
                image_path = f'static/{time.strftime("%Y%m%d_%H%M%S")}_predictions.png'
                cv2.imwrite(image_path, result['image_with_predictions'])

                return jsonify({
                    'body_language_class': result['body_language_class'],
                    'body_language_prob': result['body_language_prob'],
                    'image_path': image_path
                })

        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'Error processing the image'})

# ปรับปรุง route สำหรับรับภาพจากกล้อง
@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    try:
        # Get the captured image from the file input
        file = request.files['file']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        result = process_image(image)

        if result:
            # Save the image with predictions
            image_path = f'static/{time.strftime("%Y%m%d_%H%M%S")}_predictions.png'
            cv2.imwrite(image_path, result['image_with_predictions'])

            return jsonify({
                'body_language_class': result['body_language_class'],
                'body_language_prob': result['body_language_prob'],
                'image_path': image_path
            })

    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({'error': 'Error processing the image'})


if __name__ == '__main__':
    app.run(debug=True)
