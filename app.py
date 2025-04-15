from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, flash
import threading
import cv2
import time
import logging
import os
from werkzeug.utils import secure_filename
from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import numpy as np
from eye_tracker import EyeTracker
from head_detector import HeadDetector
from faceverifier import FaceVerifier
from audio_detector import AudioDetector
from object_detector import ObjectDetector

# Configure logging - write to file only with minimal output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("eye_tracking.log"),
    ]
)
logger = logging.getLogger("eye_tracker")

# Create a shared video capture for all trackers
cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open video capture")
except Exception as e:
    logger.error(f"Error initializing camera: {e}")

# Initialize core components
audio_detector = AudioDetector()
eye_tracker = None
head_detector = None
face_verifier = None

app = Flask(__name__)
object_detector = None
app.secret_key = 'your_secret_key_here'  # For flash messages

# Configure upload folder for passport photos
UPLOAD_FOLDER = 'passport_photos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Database path
DB_PATH = 'face_database.db'


# Initialize components after the camera is available
if cap is not None and cap.isOpened():
    eye_tracker = EyeTracker(video_source=cap)
    head_detector = HeadDetector(video_source=cap)
    face_verifier = FaceVerifier(db_path=DB_PATH, photos_dir=UPLOAD_FOLDER)
    object_detector = ObjectDetector(video_source=cap, confidence_threshold=0.45)
else:
    logger.error("Camera not available, components could not be initialized")

# Global session state
current_session = {
    'user_id': None,
    'verification_status': None,
    'verification_confidence': 0.0,
    'session_id': None,
    'is_verified': False,
    'last_verification_time': None,
    'verification_checks': 0,
    'verification_mode': False,  # New field: indicates if we're in dedicated verification mode
    'verification_start_time': None,  # New field: when verification process started
    'verification_timeout': 20,  # New field: timeout in seconds for verification
    'verification_complete': False,  # New field: indicates if verification is complete
    'max_confidence': 0.0,  # New field: tracks the highest confidence score so far
    'audio_monitoring_active': False,
    'headphones_detected': False,
    'headphones_with_mic': False,
    'audio_words_detected': 0,
    'audio_sentences_detected': 0,
    'object_detection_active': True,
    'objects_detected': 0,
    'last_object_detection': None
}

# Current frame storage
current_frame = None
frame_lock = threading.Lock()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def eye_movement_detection_thread():
    """Optimized thread function for eye and head movement detection with face verification"""
    global last_status, last_metrics_log, current_frame, current_session

    if cap is None or not cap.isOpened() or eye_tracker is None or head_detector is None:
        logger.error("Cannot start detection thread - video capture or trackers not initialized")
        return

    process_every_n_frames = 2  # Process every other frame
    verify_every_n_frames = 5  # Verify face frequently in verification mode
    frame_counter = 0
    last_status = None
    last_metrics_log = time.time()
    metrics_log_interval = 60  # Log metrics every 60 seconds

    eye_tracker.start_video_capture()

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)  # Mirror flip for display
            frame_counter += 1

            if frame_counter % process_every_n_frames == 0:
                eye_frame = eye_tracker.process_frame()
                head_frame = head_detector.process_frame(frame.copy())

                object_frame = None
                if object_detector is not None and current_session['object_detection_active']:
                    object_frame = object_detector.process_frame(frame.copy())

                    metrics = object_detector.get_detection_metrics()
                    current_session['objects_detected'] = metrics['total_objects_detected']
                    if metrics['time_since_last_detection'] is not None:
                        current_session['last_object_detection'] = time.time() - metrics['time_since_last_detection']

                    if 'recent_detections' in metrics and metrics['recent_detections']:
                        most_recent = metrics['recent_detections'][-1]['object']
                        current_session['most_recent_object'] = most_recent

                    if frame_counter % 15 == 0:
                        metrics = object_detector.get_detection_metrics()
                        current_session['objects_detected'] = metrics['total_objects_detected']
                        if metrics['time_since_last_detection'] is not None:
                            current_session['last_object_detection'] = time.time() - metrics['time_since_last_detection']

                # Decide display frame priority
            
                if head_frame is not None:
                    display_frame = head_frame
                elif eye_frame is not None:
                    display_frame = eye_frame
                else:
                    display_frame = frame

                # Face Verification Logic
                if face_verifier is not None and current_session['verification_mode'] and not current_session['verification_complete']:
                    current_time = time.time()

                    if current_session['verification_start_time'] is None:
                        current_session['verification_start_time'] = current_time

                    elapsed_time = current_time - current_session['verification_start_time']
                    time_remaining = max(0, current_session['verification_timeout'] - elapsed_time)

                    countdown_text = f"Verifying: {time_remaining:.1f}s remaining"
                    cv2.putText(display_frame, countdown_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if frame_counter % verify_every_n_frames == 0 and current_session['user_id'] is not None:
                        is_match, result, confidence = face_verifier.verify_face(frame, current_session['user_id'])

                        current_session['verification_checks'] += 1
                        current_session['verification_status'] = "Verified" if is_match else "Failed"
                        current_session['verification_confidence'] = confidence
                        current_session['last_verification_time'] = time.time()

                        if confidence > current_session['max_confidence']:
                            current_session['max_confidence'] = confidence

                        if confidence >= 0.85:
                            current_session['is_verified'] = True
                            current_session['verification_complete'] = True
                            logger.info(f"User {current_session['user_id']} verified successfully with high confidence: {confidence}")

                        verification_text = f"Match: {is_match}, Confidence: {confidence*100:.1f}%"
                        color = (0, 255, 0) if is_match else (0, 0, 255)
                        cv2.putText(display_frame, verification_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if elapsed_time >= current_session['verification_timeout']:
                        current_session['verification_complete'] = True

                        if current_session['max_confidence'] >= 0.70:
                            current_session['is_verified'] = True
                            current_session['verification_confidence'] = current_session['max_confidence']
                            logger.info(f"User {current_session['user_id']} verified based on maximum confidence: {current_session['max_confidence']}")
                        else:
                            current_session['is_verified'] = False
                            logger.warning(f"Verification failed for user {current_session['user_id']} - timeout reached with max confidence: {current_session['max_confidence']}")

                elif face_verifier is not None and not current_session['verification_mode'] and current_session['user_id'] is not None and frame_counter % 15 == 0:
                    is_match, result, confidence = face_verifier.verify_face(frame, current_session['user_id'])
                    current_session['verification_confidence'] = confidence
                    current_session['last_verification_time'] = time.time()
                    current_session['verification_checks'] += 1

                    if current_session['is_verified'] and confidence < 0.60:
                        current_session['is_verified'] = False
                        logger.warning(f"User {current_session['user_id']} verification lost - confidence dropped to {confidence}")

                with frame_lock:
                    current_frame = display_frame.copy()

                current_status = eye_tracker.get_current_gaze_status()
                if current_status != last_status:
                    last_status = current_status
                    logger.info(f"Gaze change: {current_status}")

                current_time = time.time()
                if current_time - last_metrics_log >= metrics_log_interval:
                    if eye_tracker is not None:
                        eye_metrics = eye_tracker.get_reading_metrics()
                        logger.info(f"Reading metrics: {eye_metrics}")
                    if head_detector is not None:
                        head_metrics = head_detector.get_head_tracking_metrics()
                        logger.info(f"Head tracking metrics: {head_metrics}")
                    last_metrics_log = current_time

        except Exception as e:
            logger.error(f"Error in detection thread: {e}")
            time.sleep(0.5)

        time.sleep(0.010)

        
def delayed_audio_start():
    """Start audio monitoring after a short delay to ensure system is ready"""
    time.sleep(5)  # Wait 5 seconds after system startup
    if audio_detector is not None:
        logger.info("Trying to start audio monitoring automatically")
        
        # Get diagnostic information
        devices = audio_detector.list_available_devices()
        logger.info(f"Available audio devices: {devices}")
        
        # Test the microphone first
        mic_test = audio_detector.test_microphone(duration=2)
        logger.info(f"Microphone test results: {mic_test}")
        
        # Try to start audio monitoring
        success = audio_detector.start_audio_monitoring()
        if success:
            logger.info("Audio monitoring started automatically")
        else:
            logger.warning("Failed to start audio monitoring automatically - Check microphone availability")
            
            # Try to debug audio levels
            logger.info("Attempting to debug audio levels...")
            audio_detector.debug_audio_levels(duration=5)
    else:
        logger.error("Audio detector not initialized, cannot start audio monitoring")

def auto_audio_monitoring_thread():
    while True:
        if not current_session['audio_monitoring_active']:
            logger.info("Audio monitoring inactive - Trying to start...")
            success = audio_detector.start_audio_monitoring()
            if success:
                current_session['audio_monitoring_active'] = True
                logger.info("Audio monitoring started successfully")
        time.sleep(10)  # Retry every 10 seconds if not active

audio_monitor_thread = threading.Thread(target=auto_audio_monitoring_thread)
audio_monitor_thread.daemon = True
audio_monitor_thread.start()

        
@app.route('/start_verification/<int:user_id>')
def start_verification(user_id):
    """Route to start a dedicated verification process for a user"""
    # Reset the verification session
    current_session['user_id'] = user_id
    current_session['verification_mode'] = True
    current_session['verification_start_time'] = time.time()
    current_session['verification_complete'] = False
    current_session['is_verified'] = False
    current_session['verification_confidence'] = 0.0
    current_session['max_confidence'] = 0.0
    current_session['verification_checks'] = 0
    
    flash(f'Started verification for User {user_id}')
    return redirect(url_for('index'))

def generate():
    """Generator function for video streaming"""
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.03)
                continue
            frame_to_send = current_frame.copy()
        
        # Encode the frame in JPEG format
        (flag, encoded_image) = cv2.imencode(".jpg", frame_to_send)
        
        if not flag:
            continue
            
        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')
        
        # Small sleep to control frame rate
        time.sleep(0.03)

@app.route('/')
def index():
    """Route for home page"""
    users = []
    if face_verifier is not None:
        users = face_verifier.get_all_users()
    return render_template('index.html', users=users, current_user=current_session)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Route for user registration with passport photo"""
    if face_verifier is None:
        flash('Face verification system is not available')
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'photo' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['photo']
        name = request.form.get('name')
        email = request.form.get('email')
        
        # If user doesn't select file, browser may submit an empty file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if not name:
            flash('Name is required')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read the image for face registration
            img = cv2.imread(filepath)
            
            # Register the user
            success, result = face_verifier.register_user(name, img)
            
            if success:
                flash(f'User registered successfully with ID: {result}')
                return redirect(url_for('index'))
            else:
                # Remove the file if registration failed
                os.remove(filepath)
                flash(f'Registration failed: {result}')
                return redirect(request.url)
                
    return render_template('register.html')

@app.route('/select_user/<int:user_id>')
def select_user(user_id):
    """Route to select a user for verification"""
    # Reset the session
    current_session['user_id'] = user_id
    current_session['verification_status'] = None
    current_session['verification_confidence'] = 0.0
    current_session['session_id'] = None
    current_session['is_verified'] = False
    current_session['last_verification_time'] = None
    current_session['verification_checks'] = 0
    
    flash(f'User {user_id} selected for verification')
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    """Route for video feed"""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calibrate')
def calibrate():
    """Route to trigger calibration"""
    if eye_tracker is not None:
        eye_tracker.start_calibration()
    if head_detector is not None:
        head_detector.start_calibration()
    return "Calibration started", 200

@app.route('/status')
def get_status():
    """Route to get current gaze, head, and audio status as JSON"""
    try:
        # Initialize with default values
        gaze_status = "No data"
        head_status = "No data"

        eye_metrics = {
            "total_reading_time": 0.0,
            "reading_sessions": 0,
            "continuous_reading_time": 0.0,
            "on_screen_percentage": 0.0
        }

        head_metrics = {
            "head_direction": "unknown",
            "head_movement_count": 0,
            "straight_time_percentage": 0.0
        }

        audio_metrics = {
            "audio_monitoring_active": False,
            "headphones_detected": False,
            "headphones_with_mic": False,
            "total_words_detected": 0,
            "total_sentences": 0,
            "last_spoken_text": "",
            "last_detection_time": None,
            "most_common_words": [],
            "background_noise_level": 0
        }

        # Eye tracking data
        if eye_tracker is not None:
            try:
                gaze_status = eye_tracker.get_current_gaze_status()
                eye_metrics = eye_tracker.get_reading_metrics() or eye_metrics
            except Exception as e:
                logger.error(f"Error getting eye tracking metrics: {e}")

        # Head tracking data
        if head_detector is not None:
            try:
                head_status = head_detector.get_current_head_status()
                head_metrics = head_detector.get_head_tracking_metrics() or head_metrics
            except Exception as e:
                logger.error(f"Error getting head tracking metrics: {e}")

        # Audio monitoring data
        if audio_detector is not None:
            try:
                audio_metrics = audio_detector.get_audio_metrics() or audio_metrics
            except Exception as e:
                logger.error(f"Error getting audio metrics: {e}")

        # Determine status class based on gaze status
        status_class = "no-face"
        if "Looking at Screen" in gaze_status:
            status_class = "on-screen"
        elif "Blinking" in gaze_status:
            status_class = "blink"
        elif any(direction in gaze_status for direction in ["Looking Up", "Looking Down", "Looking Left", "Looking Right"]):
            status_class = "off-screen"

        # Face verification metrics
        verification_time_remaining = 0
        if current_session['verification_mode'] and not current_session['verification_complete'] and current_session['verification_start_time']:
            elapsed_time = time.time() - current_session['verification_start_time']
            verification_time_remaining = max(0, current_session['verification_timeout'] - elapsed_time)

        verification_info = {
            "is_verified": current_session['is_verified'],
            "user_id": current_session['user_id'],
            "verification_status": current_session['verification_status'],
            "verification_confidence": current_session['verification_confidence'],
            "max_confidence": current_session['max_confidence'],
            "last_verification_time": current_session['last_verification_time'],
            "verification_checks": current_session['verification_checks'],
            "verification_mode": current_session['verification_mode'],
            "verification_complete": current_session['verification_complete'],
            "verification_time_remaining": verification_time_remaining
        }

        # Audio status
        audio_status = {
            "audio_monitoring_active": audio_metrics.get("audio_monitoring_active", False),
            "headphones_detected": audio_metrics.get("headphones_detected", False),
            "headphones_with_mic": audio_metrics.get("headphones_with_mic", False),
            "total_words_detected": audio_metrics.get("total_words_detected", 0),
            "total_sentences": audio_metrics.get("total_sentences", 0),
            "last_spoken_text": audio_metrics.get("last_spoken_text", ""),
            "last_detection_time": audio_metrics.get("last_detection_time"),
            "most_common_words": audio_metrics.get("most_common_words", []),
            "background_noise_level": audio_metrics.get("background_noise_level", 0)
        }

        # Object detection metrics
        object_info = {
            "object_detection_active": current_session['object_detection_active'],
            "objects_detected": current_session['objects_detected'],
            "last_object_detection": current_session['last_object_detection'],
            "recent_objects": [],
            "most_common_objects": []
        }

        if object_detector is not None:
            metrics = object_detector.get_detection_metrics()
            object_info["recent_objects"] = metrics.get("recent_detections", [])
            object_info["most_common_objects"] = metrics.get("most_common_objects", [])

        return jsonify({
            "gaze_status": gaze_status,
            "head_status": head_status,
            "status_class": status_class,
            "total_reading_time": eye_metrics.get("total_reading_time", 0.0),
            "reading_sessions": eye_metrics.get("reading_sessions", 0),
            "continuous_reading_time": eye_metrics.get("continuous_reading_time", 0.0),
            "on_screen_percentage": eye_metrics.get("on_screen_percentage", 0.0),
            "head_direction": head_metrics.get("head_direction", "unknown"),
            "head_movement_count": head_metrics.get("head_movement_count", 0),
            "straight_time_percentage": head_metrics.get("straight_time_percentage", 0.0),
            "verification": verification_info,
            "audio": audio_status,
            "object_detection": object_info
        })

    except Exception as e:
        logger.error(f"Error in status API: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "status": "error"
        }), 500


@app.route('/reading_metrics')
def get_reading_metrics():
    """Route to get reading metrics as JSON"""
    if eye_tracker is None:
        return jsonify({"error": "Eye tracker not available"}), 404
    return jsonify(eye_tracker.get_reading_metrics() or {})

@app.route('/head_metrics')
def get_head_metrics():
    """Route to get head tracking metrics as JSON"""
    if head_detector is None:
        return jsonify({"error": "Head detector not available"}), 404
    return jsonify(head_detector.get_head_tracking_metrics() or {})

@app.route('/verification_metrics')
def get_verification_metrics():
    """Route to get face verification metrics as JSON"""
    if face_verifier is None:
        return jsonify({"error": "Face verifier not available"}), 404
    return jsonify(face_verifier.get_verification_metrics() or {})

@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    """Route to delete a user"""
    if face_verifier is None:
        flash('Face verification system is not available')
        return redirect(url_for('index'))
        
    success = face_verifier.delete_user(user_id)
    if success:
        flash(f'User {user_id} deleted successfully')
    else:
        flash(f'Failed to delete user {user_id}')
    return redirect(url_for('index'))

@app.route('/check_headphones')
def check_headphones():
    """Route to check if headphones with mic are detected"""
    if audio_detector is None:
        return jsonify({"error": "Audio detector not available"}), 404
        
    result = audio_detector.detect_headphones()
    
    # Update session state
    current_session['headphones_detected'] = result['headphones_detected']
    current_session['headphones_with_mic'] = result['headphones_with_mic']
    
    return jsonify(result)

# Add a route to test the microphone
@app.route('/test_microphone')
def test_microphone():
    """Route to test if the microphone is working"""
    if audio_detector is None:
        return jsonify({"error": "Audio detector not available"}), 404
        
    result = audio_detector.test_microphone(duration=3)
    return jsonify(result)

# Add a route to start audio monitoring
@app.route('/start_audio_monitoring')
def start_audio_monitoring():
    """Route to start audio monitoring"""
    if audio_detector is None:
        flash('Audio detection system is not available')
        return redirect(url_for('index'))
        
    success = audio_detector.start_audio_monitoring()
    current_session['audio_monitoring_active'] = success
    
    if success:
        flash('Audio monitoring started')
    else:
        flash('Failed to start audio monitoring')
    
    return redirect(url_for('index'))

# Add a route to stop audio monitoring
@app.route('/stop_audio_monitoring')
def stop_audio_monitoring():
    """Route to stop audio monitoring"""
    if audio_detector is None:
        flash('Audio detection system is not available')
        return redirect(url_for('index'))
        
    success = audio_detector.stop_audio_monitoring()
    current_session['audio_monitoring_active'] = not success
    
    if success:
        flash('Audio monitoring stopped')
    else:
        flash('Failed to stop audio monitoring')
    
    return redirect(url_for('index'))

# Add a route to get audio metrics
@app.route('/audio_metrics')
def get_audio_metrics():
    """Route to get audio monitoring metrics"""
    if audio_detector is None:
        return jsonify({"error": "Audio detector not available"}), 404
        
    metrics = audio_detector.get_audio_metrics()
    
    # Update session state
    current_session['audio_words_detected'] = metrics['total_words_detected']
    current_session['audio_sentences_detected'] = metrics['total_sentences']
    current_session['audio_monitoring_active'] = metrics.get("audio_monitoring_active", False)  # Important!

    return jsonify(metrics)

# Add a route to get all spoken content
@app.route('/spoken_content')
def get_spoken_content():
    """Route to get all detected spoken content"""
    if audio_detector is None:
        return jsonify({"error": "Audio detector not available"}), 404
        
    content = audio_detector.get_all_spoken_content()
    return jsonify(content)

# Add a route to save all spoken content
@app.route('/save_spoken_content')
def save_spoken_content():
    """Route to save all detected spoken content to a file"""
    if audio_detector is None:
        flash('Audio detection system is not available')
        return redirect(url_for('index'))
        
    success = audio_detector.save_spoken_content_to_file()
    
    if success:
        flash('Spoken content saved to file')
    else:
        flash('Failed to save spoken content')
    
    return redirect(url_for('index'))

@app.route('/debug_audio_levels')
def debug_audio_levels():
    """Route to debug audio levels"""
    if audio_detector is None:
        flash('Audio detection system is not available')
        return redirect(url_for('index'))
        
    success = audio_detector.debug_audio_levels(duration=10)
    
    if success:
        flash('Audio level debugging completed - check logs')
    else:
        flash('Failed to run audio level debugging')
    
    return redirect(url_for('index'))


# Add a new route to get object detection metrics
@app.route('/object_metrics')
def get_object_metrics():
    """Route to get object detection metrics as JSON"""
    if object_detector is None:
        return jsonify({"error": "Object detector not available"}), 404
    return jsonify(object_detector.get_detection_metrics())

# Add a route to get all object detections
@app.route('/object_detections')
def get_object_detections():
    """Route to get all detected objects with timestamps"""
    if object_detector is None:
        return jsonify({"error": "Object detector not available"}), 404
    return jsonify(object_detector.get_all_detections())

# Add a route to save object detections to a file
@app.route('/save_object_detections')
def save_object_detections():
    """Route to save all detected objects to a CSV file"""
    if object_detector is None:
        flash('Object detection system is not available')
        return redirect(url_for('index'))
        
    success = object_detector.save_detections_to_file("object_detections.csv")
    
    if success:
        flash('Object detections saved to file')
    else:
        flash('Failed to save object detections')
    
    return redirect(url_for('index'))

# Add a route to toggle object detection
@app.route('/toggle_object_detection')
def toggle_object_detection():
    """Route to toggle object detection on/off"""
    current_session['object_detection_active'] = not current_session['object_detection_active']
    
    status = "enabled" if current_session['object_detection_active'] else "disabled"
    flash(f'Object detection {status}')
    
    return redirect(url_for('index'))

@app.route('/generate_report')
def generate_report():
    duration = request.args.get('duration', default=10, type=int)
    
    # Collect system metrics
    eye_metrics = eye_tracker.get_reading_metrics() if eye_tracker else {}
    head_metrics = head_detector.get_head_tracking_metrics() if head_detector else {}
    audio_metrics = audio_detector.get_audio_metrics() if audio_detector else {}
    object_metrics = object_detector.get_detection_metrics() if object_detector else {}
    verification_info = {
        "user_id": current_session['user_id'],
        "verified": current_session['is_verified'],
        "confidence": current_session['verification_confidence'],
        "max_confidence": current_session['max_confidence'],
        "checks": current_session['verification_checks']
    }

    # Create PDF
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, f"Cheating Detection Report - Last {duration} Minutes")
    y -= 30

    def draw_section(title, data):
        nonlocal y
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, y, title)
        y -= 20
        p.setFont("Helvetica", 10)
        for key, value in data.items():
            p.drawString(60, y, f"{key}: {value}")
            y -= 15
            if y < 50:
                p.showPage()
                y = height - 50

    draw_section("Face Verification Info", verification_info)
    draw_section("Eye Tracking Metrics", eye_metrics)
    draw_section("Head Tracking Metrics", head_metrics)
    draw_section("Audio Metrics", audio_metrics)
    draw_section("Object Detection Metrics", object_metrics)

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="cheating_detection_report.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    # Start a thread to run the eye and head movement detection
    t = threading.Thread(target=eye_movement_detection_thread)
    t.daemon = True
    t.start()
    
    logger.info("Eye and head tracking system started")
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)