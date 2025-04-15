import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import threading
import time

class EyeTracker:
    def __init__(self, video_source=None):
        
        self.frame_counter = 0
        self.process_every_n_frames = 3
        # Keep existing initialization but adjust these parameters:
        self.CALIBRATION_FRAMES = 90  # Increase for better calibration
        self.SMOOTHING_WINDOW = 5  # Reduced for less lag
        
        # Fix vertical inversion by swapping these thresholds
        self.UP_THRESHOLD_MULTIPLIER = 0.9  # Swapped from 0.8
        self.DOWN_THRESHOLD_MULTIPLIER = 0.8  # Swapped from 0.9
        
        # Add performance optimization flag
        self.process_every_n_frames = 2
        self.frame_counter = 0
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define better eye landmark indices - using more points for better accuracy
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Define iris landmarks
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        # Updated thresholds with increased vertical sensitivity
        self.SCREEN_THRESHOLD_X = 0.08
        self.SCREEN_THRESHOLD_Y = 0.05  # Reduced to increase vertical sensitivity
        self.UP_THRESHOLD_MULTIPLIER = 0.8  # Specific multiplier for upward gaze
        self.DOWN_THRESHOLD_MULTIPLIER = 0.9
        self.CALIBRATION_FRAMES = 60
        
        # Advanced smoothing for stability
        self.SMOOTHING_WINDOW = 8
        self.left_gaze_history = deque(maxlen=self.SMOOTHING_WINDOW)
        self.right_gaze_history = deque(maxlen=self.SMOOTHING_WINDOW)
        self.calibration_complete = False
        self.calibration_data = []
        
        # Reference data (center position)
        self.reference_x, self.reference_y = 0, 0
        
        # Track vertical range for adaptive thresholds
        self.min_y_value = float('inf')
        self.max_y_value = float('-inf')
        
        # Current gaze status
        self.current_gaze_status = "Not initialized"
        
        # Video capture and frame variables
        self.cap = video_source  # Store the provided video source
        self.output_frame = None
        self.lock = threading.Lock()
        self.frame_count = 0
        
        # Reading tracking variables
        self.on_screen_start_time = None
        self.reading_time = 0
        self.continuous_reading_threshold = 2.0  # seconds
        self.is_currently_reading = False
        self.total_reading_sessions = 0
        self.total_reading_time = 0
        self.last_status_change_time = time.time()
        self.reading_stats = {
            "continuous_reading_time": 0,
            "reading_sessions": 0,
            "total_reading_time": 0,
            "on_screen_percentage": 0,
            "tracking_start_time": time.time()
        }
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.blink_rate = 0  # blinks per minute
        
        
        
    def start_video_capture(self):
        """Initialize and start the webcam capture"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        
    def stop_video_capture(self):
        """Stop and release the webcam capture"""
        if self.cap is not None:
            self.cap.release()
    
    def get_eye_mesh(self, landmarks, eye_points, w, h):
        """Extract eye mesh coordinates"""
        return np.array([(landmarks[p].x * w, landmarks[p].y * h) for p in eye_points])
    
    def get_iris_position(self, landmarks, iris_points, w, h):
        """Calculate iris center position"""
        iris_points = np.array([(landmarks[p].x * w, landmarks[p].y * h) for p in iris_points])
        center = np.mean(iris_points, axis=0)
        return int(center[0]), int(center[1])
    
    def calculate_eye_aspect_ratio(self, eye_mesh):
        """Calculate eye aspect ratio to detect blinks"""
        vertical_points = [eye_mesh[1], eye_mesh[5], eye_mesh[2], eye_mesh[4]]
        horizontal_points = [eye_mesh[0], eye_mesh[3]]
        
        v1 = np.linalg.norm(vertical_points[0] - vertical_points[1])
        v2 = np.linalg.norm(vertical_points[2] - vertical_points[3])
        
        h = np.linalg.norm(horizontal_points[0] - horizontal_points[1])
        
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def calculate_relative_iris_position(self, iris_center, eye_mesh):
        """Calculate iris position relative to eye corners with improved vertical sensitivity"""
        x_min = np.min(eye_mesh[:, 0])
        x_max = np.max(eye_mesh[:, 0])
        
        # Use upper and lower eyelid landmarks for better vertical tracking
        y_values = eye_mesh[:, 1]
        sorted_y = np.sort(y_values)
        
        # Use 20% of points from top and bottom for better eyelid detection
        num_points = len(y_values)
        top_points = sorted_y[:max(1, num_points // 5)]
        bottom_points = sorted_y[-max(1, num_points // 5):]
        
        y_min = np.mean(top_points)  # Upper eyelid average
        y_max = np.mean(bottom_points)  # Lower eyelid average
        
        sorted_x_mesh = eye_mesh[eye_mesh[:, 0].argsort()]
        left_corner_x = np.mean(sorted_x_mesh[:2, 0])  # Average of leftmost points
        right_corner_x = np.mean(sorted_x_mesh[-2:, 0])  # Average of rightmost points
        
        eye_width = max(1.0, x_max - x_min)
        eye_height = max(1.0, y_max - y_min)
        
        eye_center_x = (x_min + x_max) / 2
        eye_center_y = (y_min + y_max) / 2
        
        # Increase horizontal sensitivity
        rel_x = 2.5 * (iris_center[0] - eye_center_x) / eye_width  # Increased multiplier for horizontal
        rel_y = 2 * (iris_center[1] - eye_center_y) / eye_height
        
        # Track vertical range
        self.min_y_value = min(self.min_y_value, rel_y)
        self.max_y_value = max(self.max_y_value, rel_y)
        
        return rel_x, rel_y
    
    def smooth_gaze_direction(self, rel_x, rel_y, history):
        """Apply optimized smoothing to gaze direction"""
        history.append((rel_x, rel_y))
        
        if len(history) > 2:  # Reduced from 3 for faster response
            x_values = [p[0] for p in history]
            y_values = [p[1] for p in history]
            
            # Use weighted average instead of complex filtering
            weights = np.linspace(0.5, 1.0, len(history))
            weights = weights / np.sum(weights)
            
            avg_x = np.sum([p[0] * w for p, w in zip(history, weights)])
            avg_y = np.sum([p[1] * w for p, w in zip(history, weights)])
            
            return avg_x, avg_y
        
        return rel_x, rel_y
    
    def calibrate_gaze(self, calibration_data):
        """Calculate reference points and thresholds from calibration data"""
        if len(calibration_data) < self.CALIBRATION_FRAMES * 0.8:
            return None, None, self.SCREEN_THRESHOLD_X, self.SCREEN_THRESHOLD_Y
        
        x_values = [d[0] for d in calibration_data]
        y_values = [d[1] for d in calibration_data]
        
        x_median = np.median(x_values)
        y_median = np.median(y_values)
        x_std = np.std(x_values)
        y_std = np.std(y_values)
        
        filtered_x = [x for x in x_values if abs(x - x_median) < 2 * x_std]
        filtered_y = [y for y in y_values if abs(y - y_median) < 2 * y_std]
        
        ref_x = np.mean(filtered_x)
        ref_y = np.mean(filtered_y)
        
        std_x = np.std(filtered_x)
        std_y = np.std(filtered_y)
        
        # Adjust thresholds to be more sensitive for vertical movement
        threshold_x = max(0.05, std_x * 1.0)
        threshold_y = max(0.05, std_y * 1.1)  # Reduced multiplier for vertical
        
        return ref_x, ref_y, threshold_x, threshold_y
    
    def classify_gaze(self, rel_x, rel_y, ref_x, ref_y, threshold_x, threshold_y):
        """Determine gaze direction and confidence with improved up detection"""
        dx = rel_x - ref_x
        dy = rel_y - ref_y
        
        # Decrease horizontal threshold to make left/right detection more sensitive
        adjusted_threshold_x = threshold_x * 0.8  # Make horizontal detection more sensitive
        
        # Vertical range adaptive adjustment - if we detect significant range, adjust thresholds
        vertical_range = self.max_y_value - self.min_y_value
        if vertical_range > 0.1 and self.calibration_complete:
            up_threshold = threshold_y * self.UP_THRESHOLD_MULTIPLIER
            down_threshold = threshold_y * self.DOWN_THRESHOLD_MULTIPLIER
        else:
            up_threshold = threshold_y * self.UP_THRESHOLD_MULTIPLIER
            down_threshold = threshold_y * self.DOWN_THRESHOLD_MULTIPLIER
        
        # Check directions with priority to vertical
        if dy < -up_threshold:
            confidence = min(100, int(abs(dy / up_threshold) * 100))
            return "looking_down", f"Looking Down ({confidence}% confidence)"  # Changed from Up to Down
        elif dy > down_threshold:
            confidence = min(100, int(abs(dy / down_threshold) * 100))
            return "looking_up", f"Looking Up ({confidence}% confidence)"  # Changed from Down to Up
        elif dx < -adjusted_threshold_x:
            confidence = min(100, int(abs(dx / adjusted_threshold_x) * 100))
            return "looking_left", f"Looking Left ({confidence}% confidence)"
        elif dx > -adjusted_threshold_x:
            confidence = min(100, int(abs(dx / adjusted_threshold_x) * 100))
            return "looking_right", f"Looking Right ({confidence}% confidence)"
        
        on_screen_confidence = 100 - max(
            0, 
            int(max(
                abs(dx / threshold_x), 
                abs(dy / up_threshold) if dy < 0 else abs(dy / down_threshold)
            ) * 100)
        )
        return "looking_at_screen", f"Looking at Screen ({on_screen_confidence}% confidence)"
    
    def start_calibration(self):
        """Start calibration process"""
        self.calibration_complete = False
        self.calibration_data = []
        self.frame_count = 0
        # Reset vertical range tracking
        self.min_y_value = float('inf')
        self.max_y_value = float('-inf')
        # Reset reading tracking
        self.on_screen_start_time = None
        self.reading_time = 0
        self.reading_stats = {
            "continuous_reading_time": 0,
            "reading_sessions": 0,
            "total_reading_time": 0,
            "on_screen_percentage": 0,
            "tracking_start_time": time.time()
        }
    
    def update_reading_metrics(self, gaze_code):
        """Update reading metrics based on current gaze status"""
        current_time = time.time()
        elapsed = current_time - self.last_status_change_time
        
        # Track on-screen time for reading
        if gaze_code == "looking_at_screen":
            if self.on_screen_start_time is None:
                # Just started looking at the screen
                self.on_screen_start_time = current_time
            
            # Check if we've been looking at the screen continuously for the threshold time
            continuous_time = current_time - self.on_screen_start_time
            
            # Update the longest continuous reading session
            self.reading_stats["continuous_reading_time"] = max(
                self.reading_stats["continuous_reading_time"], 
                continuous_time
            )
            
            # If we weren't reading before but now have been on screen for threshold time
            if not self.is_currently_reading and continuous_time >= self.continuous_reading_threshold:
                self.is_currently_reading = True
                self.reading_stats["reading_sessions"] += 1
            
            # Add to total reading time if we're in a reading session
            if self.is_currently_reading:
                self.reading_stats["total_reading_time"] += elapsed
        else:
            # No longer looking at screen
            self.on_screen_start_time = None
            self.is_currently_reading = False
        
        # Calculate on-screen percentage
        total_tracking_time = current_time - self.reading_stats["tracking_start_time"]
        if total_tracking_time > 0:
            self.reading_stats["on_screen_percentage"] = (
                self.reading_stats["total_reading_time"] / total_tracking_time * 100
            )
        
        # Update blink metrics if this is a blink
        if gaze_code == "blinking":
            self.blink_count += 1
            elapsed_minutes = (current_time - self.reading_stats["tracking_start_time"]) / 60.0
            if elapsed_minutes > 0:
                self.blink_rate = self.blink_count / elapsed_minutes
        
        self.last_status_change_time = current_time
    
    def process_frame(self):
        """Process a single frame with performance optimization"""
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            return None
        
        if self.cap is None:
            self.start_video_capture()
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        frame = cv2.flip(frame, 1)  # Mirror flip for intuitive display
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face landmarks
        results = self.face_mesh.process(rgb_frame)
        gaze_status = "No Face Detected"
        gaze_code = "no_face"
        
        # Default state
        avg_rel_x, avg_rel_y = 0, 0
        left_rel_x, left_rel_y = 0, 0
        right_rel_x, right_rel_y = 0, 0
        left_ear, right_ear = 0, 0
        
        # Create a clean frame without any markers
        clean_frame = frame.copy()
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Extract eye meshes
                left_eye_mesh = self.get_eye_mesh(landmarks.landmark, self.LEFT_EYE, w, h)
                right_eye_mesh = self.get_eye_mesh(landmarks.landmark, self.RIGHT_EYE, w, h)
                
                # Get iris positions
                left_iris_x, left_iris_y = self.get_iris_position(landmarks.landmark, self.LEFT_IRIS, w, h)
                right_iris_x, right_iris_y = self.get_iris_position(landmarks.landmark, self.RIGHT_IRIS, w, h)
                
                # Calculate eye aspect ratios to detect blinks
                left_ear = self.calculate_eye_aspect_ratio(left_eye_mesh)
                right_ear = self.calculate_eye_aspect_ratio(right_eye_mesh)
                
                # Skip gaze detection during blinks
                if left_ear < 0.15 or right_ear < 0.15:
                    gaze_code = "blinking"
                    gaze_status = "Blinking"
                else:
                    # Calculate relative iris positions
                    left_rel_x, left_rel_y = self.calculate_relative_iris_position((left_iris_x, left_iris_y), left_eye_mesh)
                    right_rel_x, right_rel_y = self.calculate_relative_iris_position((right_iris_x, right_iris_y), right_eye_mesh)
                    
                    # Apply smoothing
                    left_rel_x, left_rel_y = self.smooth_gaze_direction(left_rel_x, left_rel_y, self.left_gaze_history)
                    right_rel_x, right_rel_y = self.smooth_gaze_direction(right_rel_x, right_rel_y, self.right_gaze_history)
                    
                    # Average of both eyes for final gaze direction
                    avg_rel_x = (left_rel_x + right_rel_x) / 2
                    avg_rel_y = (left_rel_y + right_rel_y) / 2
                    
                    # During calibration, collect data when user is looking at screen
                    if not self.calibration_complete:
                        self.frame_count += 1
                        
                        self.calibration_data.append((avg_rel_x, avg_rel_y))
                        
                        if self.frame_count >= self.CALIBRATION_FRAMES:
                            # Calculate reference points and thresholds
                            ref_x, ref_y, threshold_x, threshold_y = self.calibrate_gaze(self.calibration_data)
                            if ref_x is not None:
                                self.reference_x, self.reference_y = ref_x, ref_y
                                self.SCREEN_THRESHOLD_X, self.SCREEN_THRESHOLD_Y = threshold_x, threshold_y
                                self.calibration_complete = True
                            else:
                                # Reset calibration if it failed
                                self.frame_count = 0
                                self.calibration_data = []
                    else:
                        # Classify gaze without debug info
                        gaze_code, gaze_status = self.classify_gaze(
                            avg_rel_x, avg_rel_y, 
                            self.reference_x, self.reference_y, 
                            self.SCREEN_THRESHOLD_X, self.SCREEN_THRESHOLD_Y
                        )
        
        # Update reading metrics if calibration is complete
        if self.calibration_complete:
            self.update_reading_metrics(gaze_code)
        
        # Store current gaze status (without emojis for API)
        self.current_gaze_status = gaze_status
        
        # Update the output frame - use the clean frame without any markers
        with self.lock:
            self.output_frame = clean_frame.copy()
            
        return clean_frame
    
    def run_tracking(self):
        """Main loop for continuous tracking"""
        self.start_video_capture()
        while True:
            self.process_frame()
    
    def get_output_frame(self):
        """Get the current processed frame"""
        with self.lock:
            if self.output_frame is None:
                return None
            return self.output_frame.copy()
    
    def get_current_gaze_status(self):
        """Get the current gaze status for external use"""
        return self.current_gaze_status
    
    def get_reading_metrics(self):
        """Get reading metrics for external use"""
        return self.reading_stats