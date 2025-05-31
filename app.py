import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import time
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import base64
import io
from PIL import Image
import threading
import queue
import sqlite3
import hashlib
from datetime import datetime, timedelta
from functools import wraps
import os
import jwt
from flask import Flask, request, jsonify, session, render_template

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = 'Secretkey'  # Change this in production

# Database setup
def init_db():
    conn = sqlite3.connect('pose_detection.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # User sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_date DATE DEFAULT CURRENT_DATE,
            yoga_sessions INTEGER DEFAULT 0,
            gym_sessions INTEGER DEFAULT 0,
            total_reps INTEGER DEFAULT 0,
            session_data TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user_id = data['user_id']
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

class UnifiedPoseDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Load yoga model with better error handling
        self.yoga_model = None
        self.load_yoga_model()
        
        # Confidence threshold for yoga pose detection (50%)
        self.confidence_threshold = 0.5
        
        # Gym exercise counters and stages
        self.gym_counters = {
            'bicep_curls': 0,
            'squats': 0,
            'push_ups': 0,
            'shoulder_press': 0,
            'lateral_raises': 0,
            'lunges': 0,
            'pull_ups': 0,
            'overhead_press': 0
        }
        
        self.gym_stages = {
            'bicep_curls': None,
            'squats': None,
            'push_ups': None,
            'shoulder_press': None,
            'lateral_raises': None,
            'lunges': None,
            'pull_ups': None,
            'overhead_press': None
        }
        
        self.current_exercise = 'bicep_curls'
        self.mode = 'yoga'  # 'yoga' or 'gym'
    
    def load_yoga_model(self):
        """Load yoga model with comprehensive error handling"""
        model_filename = 'Yogaposes02.pkl'
        
        # Print current working directory for debugging
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")
        
        # List files in current directory to verify file existence
        files_in_dir = os.listdir(current_dir)
        print(f"Files in current directory: {files_in_dir}")
        
        # Check if file exists
        if not os.path.exists(model_filename):
            print(f"ERROR: File '{model_filename}' not found in {current_dir}")
            print("Please ensure the file is in the same directory as your Python script.")
            return
        
        # Check file permissions
        if not os.access(model_filename, os.R_OK):
            print(f"ERROR: File '{model_filename}' exists but is not readable. Check file permissions.")
            return
        
        # Check file size
        file_size = os.path.getsize(model_filename)
        print(f"File size: {file_size} bytes")
        
        if file_size == 0:
            print(f"ERROR: File '{model_filename}' is empty.")
            return
        
        # Try to load the model
        try:
            print(f"Attempting to load '{model_filename}'...")
            with open(model_filename, 'rb') as f:
                self.yoga_model = pickle.load(f)
            print("✓ Yoga model loaded successfully!")
            
            # Verify the model has required methods
            if hasattr(self.yoga_model, 'predict') and hasattr(self.yoga_model, 'predict_proba'):
                print("✓ Model has required prediction methods")
            else:
                print("WARNING: Model may not have required methods (predict, predict_proba)")
                
        except pickle.UnpicklingError as e:
            print(f"ERROR: Failed to unpickle the file. The file may be corrupted or created with incompatible Python version.")
            print(f"Pickle error: {e}")
            
        except Exception as e:
            print(f"ERROR: Unexpected error loading yoga model: {e}")
            print(f"Error type: {type(e).__name__}")
    
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold for yoga pose detection"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            return True
        return False
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def get_landmark_coords(self, landmarks, landmark_name):
        """Get x,y coordinates of a landmark"""
        landmark = landmarks[landmark_name.value]
        return [landmark.x, landmark.y]
    
    def detect_yoga_pose(self, results):
        """Detect yoga pose using the trained model with confidence threshold"""
        if not results.pose_landmarks or not results.face_landmarks:
            return "No Pose Detected", 0.0
            
        try:
            # Extract pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                    for landmark in pose]).flatten())
            
            # Extract face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                    for landmark in face]).flatten())
            
            # Combine features
            row = pose_row + face_row
            
            if self.yoga_model:
                # Make prediction
                X = pd.DataFrame([row])
                pose_class = self.yoga_model.predict(X)[0]
                pose_prob = self.yoga_model.predict_proba(X)[0]
                confidence = np.max(pose_prob)
                
                # Check if confidence meets threshold
                if confidence >= self.confidence_threshold:
                    return pose_class, confidence
                else:
                    return "Low Confidence Detection", confidence
            else:
                # Dummy data if model not available
                dummy_poses = ['Tree Pose', 'Warrior I', 'Downward Dog', 'Mountain Pose', 'Triangle Pose']
                confidence = np.random.uniform(0.3, 0.95)
                pose = np.random.choice(dummy_poses)
                
                if confidence >= self.confidence_threshold:
                    return pose, confidence
                else:
                    return "Low Confidence Detection", confidence
                
        except Exception as e:
            print(f"Error in yoga pose detection: {e}")
            return "Error", 0.0

    
    def detect_bicep_curls(self, landmarks):
        """Detect bicep curls using left arm"""
        try:
            shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
            wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
            
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            if angle > 160:
                self.gym_stages['bicep_curls'] = "down"
            if angle < 30 and self.gym_stages['bicep_curls'] == 'down':
                self.gym_stages['bicep_curls'] = "up"
                self.gym_counters['bicep_curls'] += 1
                
            return angle, elbow
        except:
            return None, None
    
    def detect_squats(self, landmarks):
        """Detect squats using hip-knee-ankle angle"""
        try:
            hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
            knee = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
            ankle = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
            
            angle = self.calculate_angle(hip, knee, ankle)
            
            if angle > 160:
                self.gym_stages['squats'] = "up"
            if angle < 90 and self.gym_stages['squats'] == 'up':
                self.gym_stages['squats'] = "down"
                self.gym_counters['squats'] += 1
                
            return angle, knee
        except:
            return None, None
    
    def detect_push_ups(self, landmarks):
        """Detect push-ups using elbow angle"""
        try:
            shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
            wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
            
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            if angle < 90:
                self.gym_stages['push_ups'] = "down"
            if angle > 160 and self.gym_stages['push_ups'] == 'down':
                self.gym_stages['push_ups'] = "up"
                self.gym_counters['push_ups'] += 1
                
            return angle, elbow
        except:
            return None, None
    
    def detect_shoulder_press(self, landmarks):
        """Detect shoulder press using shoulder-elbow-wrist angle"""
        try:
            shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
            wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
            
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            if wrist[1] < elbow[1]:
                if angle > 160:
                    self.gym_stages['shoulder_press'] = "up"
                if angle < 90 and self.gym_stages['shoulder_press'] == 'up':
                    self.gym_stages['shoulder_press'] = "down"
                    self.gym_counters['shoulder_press'] += 1
                    
            return angle, elbow
        except:
            return None, None
    
    def detect_lateral_raises(self, landmarks):
        """Detect lateral raises using shoulder angle"""
        try:
            shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
            hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
            
            angle = self.calculate_angle(hip, shoulder, elbow)
            
            if angle < 30:
                self.gym_stages['lateral_raises'] = "down"
            if angle > 80 and self.gym_stages['lateral_raises'] == 'down':
                self.gym_stages['lateral_raises'] = "up"
                self.gym_counters['lateral_raises'] += 1
                
            return angle, shoulder
        except:
            return None, None
    
    def detect_lunges(self, landmarks):
        """Detect lunges using front leg knee angle"""
        try:
            hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
            knee = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE)
            ankle = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
            
            angle = self.calculate_angle(hip, knee, ankle)
            
            if angle > 160:
                self.gym_stages['lunges'] = "up"
            if angle < 90 and self.gym_stages['lunges'] == 'up':
                self.gym_stages['lunges'] = "down"
                self.gym_counters['lunges'] += 1
                
            return angle, knee
        except:
            return None, None
    
    def detect_pull_ups(self, landmarks):
        """Detect pull-ups using elbow angle"""
        try:
            shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
            wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
            
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            if wrist[1] > elbow[1] and angle > 160:
                self.gym_stages['pull_ups'] = "down"
            if wrist[1] < elbow[1] and angle < 60 and self.gym_stages['pull_ups'] == 'down':
                self.gym_stages['pull_ups'] = "up"
                self.gym_counters['pull_ups'] += 1
                
            return angle, elbow
        except:
            return None, None
    
    def detect_overhead_press(self, landmarks):
        """Detect overhead press using shoulder-elbow-wrist alignment"""
        try:
            shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
            wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
            
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            if angle < 90:
                self.gym_stages['overhead_press'] = "down"
            if angle > 160 and wrist[1] < shoulder[1] and self.gym_stages['overhead_press'] == 'down':
                self.gym_stages['overhead_press'] = "up"
                self.gym_counters['overhead_press'] += 1
                
            return angle, elbow
        except:
            return None, None
    
    def process_gym_exercise(self, landmarks):
        """Process current gym exercise"""
        exercise_functions = {
            'bicep_curls': self.detect_bicep_curls,
            'squats': self.detect_squats,
            'push_ups': self.detect_push_ups,
            'shoulder_press': self.detect_shoulder_press,
            'lateral_raises': self.detect_lateral_raises,
            'lunges': self.detect_lunges,
            'pull_ups': self.detect_pull_ups,
            'overhead_press': self.detect_overhead_press
        }
        
        if self.current_exercise in exercise_functions:
            return exercise_functions[self.current_exercise](landmarks)
        return None, None
    
    def process_frame(self, image):
        """Process a single frame and return detection results"""
        results = {}
        
        if self.mode == 'yoga':
            # Use holistic model for yoga
            with self.mp_holistic.Holistic(
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5
            ) as holistic:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                holistic_results = holistic.process(image_rgb)
                
                pose_class, confidence = self.detect_yoga_pose(holistic_results)
                
                results = {
                    'mode': 'yoga',
                    'pose_class': pose_class,
                    'confidence': float(confidence),
                    'confidence_threshold': self.confidence_threshold,
                    'meets_threshold': confidence >= self.confidence_threshold,
                    'landmarks': holistic_results.pose_landmarks is not None,
                    'model_loaded': self.yoga_model is not None
                }
                
        else:  # gym mode
            # Use pose model for gym exercises
            with self.mp_pose.Pose(
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5
            ) as pose:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(image_rgb)
                
                angle = None
                if pose_results.pose_landmarks:
                    angle, _ = self.process_gym_exercise(pose_results.pose_landmarks.landmark)
                
                results = {
                    'mode': 'gym',
                    'exercise': self.current_exercise,
                    'count': self.gym_counters[self.current_exercise],
                    'stage': self.gym_stages[self.current_exercise],
                    'angle': float(angle) if angle else None,
                    'total_reps': sum(self.gym_counters.values()),
                    'all_counters': self.gym_counters.copy(),
                    'landmarks': pose_results.pose_landmarks is not None
                }
        
        return results

# Global detector instance
detector = UnifiedPoseDetector()

# Authentication routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    
    # Hash password
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    conn = sqlite3.connect('pose_detection.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                      (username, email, password_hash))
        conn.commit()
        user_id = cursor.lastrowid
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user_id,
            'username': username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.secret_key, algorithm='HS256')
        
        conn.close()
        return jsonify({
            'success': True,
            'token': token,
            'user': {'id': user_id, 'username': username, 'email': email}
        })
        
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Username or email already exists'}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    conn = sqlite3.connect('pose_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, username, email FROM users WHERE username = ? AND password = ?',
                   (username, password_hash))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        # Generate JWT token
        token = jwt.encode({
            'user_id': user[0],
            'username': user[1],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {'id': user[0], 'username': user[1], 'email': user[2]}
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
@token_required
def logout(current_user_id):
    return jsonify({'success': True, 'message': 'Logged out successfully'})

# Protected routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'message': 'Pose detection API is running',
        'yoga_model_loaded': detector.yoga_model is not None,
        'confidence_threshold': detector.confidence_threshold
    })

@app.route('/api/set_mode', methods=['POST'])
@token_required
def set_mode(current_user_id):
    data = request.json
    mode = data.get('mode', 'yoga')
    
    if mode in ['yoga', 'gym']:
        detector.mode = mode
        return jsonify({'success': True, 'mode': mode})
    else:
        return jsonify({'success': False, 'error': 'Invalid mode'}), 400

@app.route('/api/set_exercise', methods=['POST'])
@token_required
def set_exercise(current_user_id):
    data = request.json
    exercise = data.get('exercise')
    
    if exercise in detector.gym_counters.keys():
        detector.current_exercise = exercise
        return jsonify({'success': True, 'exercise': exercise})
    else:
        return jsonify({'success': False, 'error': 'Invalid exercise'}), 400

@app.route('/api/set_confidence_threshold', methods=['POST'])
@token_required
def set_confidence_threshold(current_user_id):
    """Set confidence threshold for yoga pose detection"""
    data = request.json
    threshold = data.get('threshold', 0.5)
    
    if detector.set_confidence_threshold(threshold):
        return jsonify({
            'success': True, 
            'threshold': threshold,
            'message': f'Confidence threshold set to {threshold*100}%'
        })
    else:
        return jsonify({
            'success': False, 
            'error': 'Invalid threshold. Must be between 0.0 and 1.0'
        }), 400

@app.route('/api/reset_counter', methods=['POST'])
@token_required
def reset_counter(current_user_id):
    data = request.json
    exercise = data.get('exercise', detector.current_exercise)
    
    if exercise == 'all':
        for key in detector.gym_counters:
            detector.gym_counters[key] = 0
            detector.gym_stages[key] = None
        return jsonify({'success': True, 'message': 'All counters reset'})
    elif exercise in detector.gym_counters:
        detector.gym_counters[exercise] = 0
        detector.gym_stages[exercise] = None
        return jsonify({'success': True, 'exercise': exercise, 'count': 0})
    else:
        return jsonify({'success': False, 'error': 'Invalid exercise'}), 400

@app.route('/api/process_frame', methods=['POST'])
@token_required
def process_frame(current_user_id):
    try:
        data = request.json
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process frame
        results = detector.process_frame(opencv_image)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_status', methods=['GET'])
@token_required
def get_status(current_user_id):
    return jsonify({
        'mode': detector.mode,
        'current_exercise': detector.current_exercise,
        'gym_counters': detector.gym_counters,
        'gym_stages': detector.gym_stages,
        'yoga_model_loaded': detector.yoga_model is not None,
        'confidence_threshold': detector.confidence_threshold
    })

# @app.route('/api/save_session', methods=['POST'])
# @token_required
# def save_session(current_user_id):
#     data = request.json
    
#     conn = sqlite3.connect('pose_detection.db')
#     cursor = conn.cursor()
#     # Get or create today's session
#     cursor.execute('''
#         INSERT OR IGNORE INTO user_sessions (user_id, yoga_sessions, gym_sessions, total_reps) 
#         VALUES (?, 0, 0, 0)
#     ''', (current_user_id,))
    
#     # Update session data
#     if detector.mode == 'yoga':
#         cursor.execute('''
#             UPDATE user_sessions 
#             SET yoga_sessions = yoga_sessions + 1 
#             WHERE user_id = ? AND session_date = CURRENT_DATE
#         ''', (current_user_id,))
#     else:
#         total_reps = sum(detector.gym_counters.values())
#         cursor.execute('''
#             UPDATE user_sessions 
#             SET gym_sessions = gym_sessions + 1, total_reps = total_reps + ?
#             WHERE user_id = ? AND session_date = CURRENT_DATE
#         ''', (total_reps, current_user_id))
    
#     conn.commit()
#     conn.close()
    
#     return jsonify({'success': True, 'message': 'Session saved'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0', port=5000)