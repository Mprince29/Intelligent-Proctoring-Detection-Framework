import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import threading

class HeadDetector:
    def __init__(self, video_source=None):
        # Initialize the face mesh detection from MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Landmark indices for head pose estimation
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.NOSE_TIP = 4
        self.NOSE_BRIDGE = 6
        self.CHIN = 152
        self.LEFT_EYE_LEFT_CORNER = 33
        self.LEFT_EYE_RIGHT_CORNER = 133
        self.RIGHT_EYE_LEFT_CORNER = 362
        self.RIGHT_EYE_RIGHT_CORNER = 263
        self.LEFT_EYEBROW_LEFT = 70
        self.RIGHT_EYEBROW_RIGHT = 300
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.FOREHEAD = 10

        # Parameters for smoothing and thresholds with improved values
        self.SMOOTHING_WINDOW = 10
        self.history = deque(maxlen=self.SMOOTHING_WINDOW)
        self.calibration_complete = False
        self.calibration_data = []
        self.CALIBRATION_FRAMES = 90
        
        # Reference data (neutral head position)
        self.reference_roll = 0
        self.reference_pitch = 0
        self.reference_yaw = 0
        self.frame_count = 0
        
        # Reduced thresholds for detecting slight head movements
        self.ROLL_THRESHOLD = 6     # Reduced from 12
        self.PITCH_THRESHOLD = 6    # Reduced from 12
        self.YAW_THRESHOLD = 8      # Reduced from 18
        
        # Add thresholds for slight movements
        self.SLIGHT_ROLL_THRESHOLD = 3
        self.SLIGHT_PITCH_THRESHOLD = 3
        self.SLIGHT_YAW_THRESHOLD = 4
        
        # Current head status
        self.current_head_status = "Not initialized"
        self.current_direction_vector = (0, 0, 0)
        
        # Use external video source if provided
        self.external_video_source = video_source
        self.output_frame = None
        self.lock = threading.Lock()
        
        # Tracking metrics
        self.head_tracking_stats = {
            "head_direction": "no_face",
            "head_direction_vector": (0, 0, 0),
            "head_movement_count": 0,
            "slight_movement_count": 0,  # New counter for slight movements
            "last_movement_time": time.time(),
            "tracking_start_time": time.time(),
            "straight_time_percentage": 0.0,
            "straight_head_time": 0.0,
            "last_status_change_time": time.time()
        }
    
    def get_face_landmarks(self, landmarks, img_width, img_height):
        """Convert landmarks to pixel coordinates"""
        points = []
        for landmark in landmarks:
            x = int(landmark.x * img_width)
            y = int(landmark.y * img_height)
            z = landmark.z
            points.append((x, y, z))
        return np.array(points, dtype=np.float32)
    
    def estimate_head_pose(self, landmarks, img_width, img_height):
        """Estimate head pose using facial landmarks with improved algorithm"""
        # Get specific landmarks for pose estimation
        nose_tip = np.array([landmarks[self.NOSE_TIP].x * img_width, 
                            landmarks[self.NOSE_TIP].y * img_height, 
                            landmarks[self.NOSE_TIP].z])
        
        nose_bridge = np.array([landmarks[self.NOSE_BRIDGE].x * img_width, 
                               landmarks[self.NOSE_BRIDGE].y * img_height, 
                               landmarks[self.NOSE_BRIDGE].z])
        
        chin = np.array([landmarks[self.CHIN].x * img_width, 
                         landmarks[self.CHIN].y * img_height, 
                         landmarks[self.CHIN].z])
        
        left_eye_left = np.array([landmarks[self.LEFT_EYE_LEFT_CORNER].x * img_width, 
                                 landmarks[self.LEFT_EYE_LEFT_CORNER].y * img_height, 
                                 landmarks[self.LEFT_EYE_LEFT_CORNER].z])
        
        left_eye_right = np.array([landmarks[self.LEFT_EYE_RIGHT_CORNER].x * img_width, 
                                  landmarks[self.LEFT_EYE_RIGHT_CORNER].y * img_height, 
                                  landmarks[self.LEFT_EYE_RIGHT_CORNER].z])
        
        right_eye_left = np.array([landmarks[self.RIGHT_EYE_LEFT_CORNER].x * img_width, 
                                  landmarks[self.RIGHT_EYE_LEFT_CORNER].y * img_height, 
                                  landmarks[self.RIGHT_EYE_LEFT_CORNER].z])
        
        right_eye_right = np.array([landmarks[self.RIGHT_EYE_RIGHT_CORNER].x * img_width, 
                                   landmarks[self.RIGHT_EYE_RIGHT_CORNER].y * img_height, 
                                   landmarks[self.RIGHT_EYE_RIGHT_CORNER].z])
        
        forehead = np.array([landmarks[self.FOREHEAD].x * img_width, 
                            landmarks[self.FOREHEAD].y * img_height, 
                            landmarks[self.FOREHEAD].z])
        
        # Calculate eye centers
        left_eye_center = (left_eye_left + left_eye_right) / 2
        right_eye_center = (right_eye_left + right_eye_right) / 2
        
        # Calculate face center more accurately using multiple points
        face_center = np.mean([nose_tip, chin, left_eye_center, right_eye_center], axis=0)
        
        # Calculate horizontal axis (vector between eye centers)
        horizontal_axis = right_eye_center - left_eye_center
        horizontal_axis = horizontal_axis / np.linalg.norm(horizontal_axis)
        
        # Calculate vertical axis (forehead to chin vector)
        vertical_axis = chin - forehead
        vertical_axis = vertical_axis / np.linalg.norm(vertical_axis)
        
        # Calculate normal vector (perpendicular to face plane, pointing outward from face)
        normal_vector = np.cross(horizontal_axis, vertical_axis)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        
        # Calculate gaze direction vector (from face center through nose tip)
        gaze_vector = nose_tip - face_center
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        
        # Invert x-axis for the gaze vector to correct for mirror-flip
        gaze_vector[0] = -gaze_vector[0]
        
        # Store direction vector for external use
        self.current_direction_vector = tuple(gaze_vector)
        
        # Calculate head rotation angles in degrees
        roll = np.degrees(np.arctan2(horizontal_axis[1], horizontal_axis[0]))
        pitch = np.degrees(np.arcsin(normal_vector[1]))
        
        # Invert yaw calculation to correct left-right direction
        yaw = -np.degrees(np.arctan2(normal_vector[0], normal_vector[2]))
        
        return roll, pitch, yaw, gaze_vector
    
    def smooth_angles(self, roll, pitch, yaw, direction_vector):
        """Apply improved smoothing to angle measurements and direction vector"""
        self.history.append((roll, pitch, yaw, direction_vector))
        
        if len(self.history) >= 5:  # Need at least 5 samples for smoothing
            # Use exponential weighted average for smoother transitions
            weights = np.exp(np.linspace(0, 1, len(self.history)))
            weights = weights / np.sum(weights)
            
            roll_values = [angle[0] for angle in self.history]
            pitch_values = [angle[1] for angle in self.history]
            yaw_values = [angle[2] for angle in self.history]
            vector_values = [angle[3] for angle in self.history]
            
            # Apply weighted averaging to angles
            smooth_roll = np.sum([r * w for r, w in zip(roll_values, weights)])
            smooth_pitch = np.sum([p * w for p, w in zip(pitch_values, weights)])
            smooth_yaw = np.sum([y * w for y, w in zip(yaw_values, weights)])
            
            # Apply weighted averaging to direction vector
            vector_x = np.sum([v[0] * w for v, w in zip(vector_values, weights)])
            vector_y = np.sum([v[1] * w for v, w in zip(vector_values, weights)])
            vector_z = np.sum([v[2] * w for v, w in zip(vector_values, weights)])
            
            # Normalize the resulting direction vector
            vec_norm = np.sqrt(vector_x**2 + vector_y**2 + vector_z**2)
            if vec_norm > 1e-6:  # Avoid division by zero
                smooth_direction = (vector_x/vec_norm, vector_y/vec_norm, vector_z/vec_norm)
            else:
                smooth_direction = (0, 0, 1)  # Default forward direction
            
            return smooth_roll, smooth_pitch, smooth_yaw, smooth_direction
        
        return roll, pitch, yaw, direction_vector
    
    def start_calibration(self):
        """Start the calibration process"""
        self.calibration_complete = False
        self.calibration_data = []
        self.frame_count = 0
        
        # Reset tracking stats
        self.head_tracking_stats = {
            "head_direction": "no_face",
            "head_direction_vector": (0, 0, 0),
            "head_movement_count": 0,
            "slight_movement_count": 0,
            "last_movement_time": time.time(),
            "tracking_start_time": time.time(),
            "straight_time_percentage": 0.0,
            "straight_head_time": 0.0,
            "last_status_change_time": time.time()
        }
    
    def calibrate_head_pose(self, calibration_data):
        """Calculate reference angles from calibration data with improved outlier rejection"""
        if len(calibration_data) < self.CALIBRATION_FRAMES * 0.7:
            return None, None, None
        
        roll_values = [d[0] for d in calibration_data]
        pitch_values = [d[1] for d in calibration_data]
        yaw_values = [d[2] for d in calibration_data]
        
        # Use more robust statistics for better outlier handling
        roll_median = np.median(roll_values)
        pitch_median = np.median(pitch_values)
        yaw_median = np.median(yaw_values)
        
        # Calculate median absolute deviation (MAD) for more robust outlier detection
        roll_mad = np.median([abs(r - roll_median) for r in roll_values])
        pitch_mad = np.median([abs(p - pitch_median) for p in pitch_values])
        yaw_mad = np.median([abs(y - yaw_median) for y in yaw_values])
        
        # Filter outliers using MAD (more robust than standard deviation)
        mad_factor = 2.5  # Equivalent to about 3 standard deviations for normal distributions
        filtered_roll = [r for r in roll_values if abs(r - roll_median) < mad_factor * roll_mad]
        filtered_pitch = [p for p in pitch_values if abs(p - pitch_median) < mad_factor * pitch_mad]
        filtered_yaw = [y for y in yaw_values if abs(y - yaw_median) < mad_factor * yaw_mad]
        
        # Use mean of inliers as reference
        ref_roll = np.mean(filtered_roll)
        ref_pitch = np.mean(filtered_pitch)
        ref_yaw = np.mean(filtered_yaw)
        
        # Adjust thresholds based on variation in the calibration data
        # Lower bounds to increase sensitivity, upper bounds prevent over-sensitivity
        self.ROLL_THRESHOLD = max(6, min(15, np.std(filtered_roll) * 2.0))
        self.PITCH_THRESHOLD = max(6, min(15, np.std(filtered_pitch) * 2.0))
        self.YAW_THRESHOLD = max(8, min(20, np.std(filtered_yaw) * 2.0))
        
        # Set slight movement thresholds to be a fraction of the main thresholds
        self.SLIGHT_ROLL_THRESHOLD = max(3, self.ROLL_THRESHOLD * 0.5)
        self.SLIGHT_PITCH_THRESHOLD = max(3, self.PITCH_THRESHOLD * 0.5)
        self.SLIGHT_YAW_THRESHOLD = max(4, self.YAW_THRESHOLD * 0.5)
        
        return ref_roll, ref_pitch, ref_yaw
    
    def classify_head_direction(self, roll, pitch, yaw, direction_vector):
        """Determine head direction based on angles and direction vector with increased sensitivity"""
        # Calculate differences from reference
        roll_diff = roll - self.reference_roll
        pitch_diff = pitch - self.reference_pitch
        yaw_diff = yaw - self.reference_yaw
        
        # Determine movement amounts
        abs_roll_diff = abs(roll_diff)
        abs_pitch_diff = abs(pitch_diff)
        abs_yaw_diff = abs(yaw_diff)
        
        # Extract direction vector components
        x, y, z = direction_vector
        
        # Increased sensitivity factors
        pitch_sensitivity = 0.9
        yaw_sensitivity = 0.9
        
        # Normalize difference values relative to thresholds for comparing between axes
        norm_roll_diff = abs_roll_diff / self.ROLL_THRESHOLD
        norm_pitch_diff = abs_pitch_diff / self.PITCH_THRESHOLD
        norm_yaw_diff = abs_yaw_diff / self.YAW_THRESHOLD
        
        # Create detailed classification with confidence levels
        
        # Check for significant pitch movement (looking up/down)
        if abs_pitch_diff > (self.PITCH_THRESHOLD * pitch_sensitivity) and (norm_pitch_diff >= norm_yaw_diff * 0.8 or abs(y) > abs(x) * 0.8):
            if pitch_diff < 0 or y < -0.1:  # Looking up
                confidence = min(100, int((max(abs_pitch_diff / self.PITCH_THRESHOLD, abs(y) * 3) * 100)))
                return "looking_up", f"Head Up ({confidence}% confidence)", direction_vector
            else:  # Looking down
                confidence = min(100, int((max(abs_pitch_diff / self.PITCH_THRESHOLD, abs(y) * 3) * 100)))
                return "looking_down", f"Head Down ({confidence}% confidence)", direction_vector
        
        # Check for significant yaw movement (looking left/right)
        elif abs_yaw_diff > (self.YAW_THRESHOLD * yaw_sensitivity) and (norm_yaw_diff >= norm_pitch_diff * 0.8 or abs(x) > abs(y) * 0.8):
            if yaw_diff < 0 or x < -0.1:  # Looking left
                confidence = min(100, int((max(abs_yaw_diff / self.YAW_THRESHOLD, abs(x) * 3) * 100)))
                return "looking_right", f"Head Right ({confidence}% confidence)", direction_vector
            else:  # Looking right
                confidence = min(100, int((max(abs_yaw_diff / self.YAW_THRESHOLD, abs(x) * 3) * 100)))
                return "looking_left", f"Head Left ({confidence}% confidence)", direction_vector
        
        # Check for slight yaw movement (slight left/right turn)
        elif abs_yaw_diff > self.SLIGHT_YAW_THRESHOLD and (norm_yaw_diff >= norm_pitch_diff * 0.7 or abs(x) > abs(y) * 0.7):
            if yaw_diff < 0 or x < -0.05:  # Slight left turn
                confidence = min(100, int((max(abs_yaw_diff / self.SLIGHT_YAW_THRESHOLD, abs(x) * 4) * 100)))
                return "slight_Right", f"Slight Right Turn ({confidence}% confidence)", direction_vector
            else:  # Slight right turn
                confidence = min(100, int((max(abs_yaw_diff / self.SLIGHT_YAW_THRESHOLD, abs(x) * 4) * 100)))
                return "slight_Left", f"Slight Left Turn ({confidence}% confidence)", direction_vector
        
        # Check for slight pitch movement (slight up/down tilt)
        elif abs_pitch_diff > self.SLIGHT_PITCH_THRESHOLD and (norm_pitch_diff >= norm_yaw_diff * 0.7 or abs(y) > abs(x) * 0.7):
            if pitch_diff < 0 or y < -0.05:  # Slight up tilt
                confidence = min(100, int((max(abs_pitch_diff / self.SLIGHT_PITCH_THRESHOLD, abs(y) * 4) * 100)))
                return "slight_up", f"Slight Up Tilt ({confidence}% confidence)", direction_vector
            else:  # Slight down tilt
                confidence = min(100, int((max(abs_pitch_diff / self.SLIGHT_PITCH_THRESHOLD, abs(y) * 4) * 100)))
                return "slight_down", f"Slight Down Tilt ({confidence}% confidence)", direction_vector
        
        # Check for significant roll movement (head tilt)
        elif abs_roll_diff > self.ROLL_THRESHOLD:
            if roll_diff < 0:  # Head tilted right
                confidence = min(100, int((abs_roll_diff / self.ROLL_THRESHOLD * 100)))
                return "head_tilted_Left", f"Head Tilted Left ({confidence}% confidence)", direction_vector
            else:  # Head tilted left
                confidence = min(100, int((abs_roll_diff / self.ROLL_THRESHOLD * 100)))
                return "head_tilted_right", f"Head Tilted Right ({confidence}% confidence)", direction_vector
        
        # Check for slight roll movement (slight head tilt)
        elif abs_roll_diff > self.SLIGHT_ROLL_THRESHOLD:
            if roll_diff < 0:  # Slight tilt right
                confidence = min(100, int((abs_roll_diff / self.SLIGHT_ROLL_THRESHOLD * 100)))
                return "slight_tilt_left", f"Slight Left Tilt ({confidence}% confidence)", direction_vector
            else:  # Slight tilt left
                confidence = min(100, int((abs_roll_diff / self.SLIGHT_ROLL_THRESHOLD * 100)))
                return "slight_tilt_right", f"Slight Right Tilt ({confidence}% confidence)", direction_vector
        
        # Calculate how centered the head is using both angles and direction vector
        angle_deviation = max(
            abs_roll_diff / self.ROLL_THRESHOLD,
            abs_pitch_diff / self.PITCH_THRESHOLD,
            abs_yaw_diff / self.YAW_THRESHOLD
        )
        
        vector_deviation = np.sqrt(x*x + y*y) / 0.3  # Normalize deviation from forward (0,0,1)
        
        # Calculate deviations relative to slight movement thresholds
        slight_angle_deviation = max(
            abs_roll_diff / self.SLIGHT_ROLL_THRESHOLD,
            abs_pitch_diff / self.SLIGHT_PITCH_THRESHOLD,
            abs_yaw_diff / self.SLIGHT_YAW_THRESHOLD
        )
        
        # If there's a very slight deviation but not enough to classify as a specific movement
        if slight_angle_deviation > 0.7 and slight_angle_deviation < 1.0:
            # Determine the most prominent slight movement
            if norm_yaw_diff > norm_pitch_diff and norm_yaw_diff > norm_roll_diff:
                direction = "Slight Horizontal" if yaw_diff > 0 else "Slight Horizontal"
                confidence = min(100, int(slight_angle_deviation * 100))
                return "slight_movement", f"{direction} ({confidence}% confidence)", direction_vector
            elif norm_pitch_diff > norm_roll_diff:
                direction = "Slight Vertical" if pitch_diff > 0 else "Slight Vertical"
                confidence = min(100, int(slight_angle_deviation * 100))
                return "slight_movement", f"{direction} ({confidence}% confidence)", direction_vector
            else:
                direction = "Slight Tilt" if roll_diff > 0 else "Slight Tilt"
                confidence = min(100, int(slight_angle_deviation * 100))
                return "slight_movement", f"{direction} ({confidence}% confidence)", direction_vector
        
        # Combine both metrics for a more robust "straight" classification
        max_deviation = max(angle_deviation, vector_deviation)
        
        straight_confidence = max(0, min(100, int((1 - max_deviation) * 100)))
        return "head_straight", f"Head Straight ({straight_confidence}% confidence)", direction_vector
    
    def update_head_metrics(self, head_code, head_status, direction_vector):
        """Update head tracking metrics with more detailed movement tracking"""
        current_time = time.time()
        elapsed = current_time - self.head_tracking_stats["last_status_change_time"]
        
        # Store previous status to detect changes
        prev_head_code = self.head_tracking_stats["head_direction"]
        
        # If direction changed
        if head_code != prev_head_code:
            # Update movement count if moving from straight to any direction
            if prev_head_code == "head_straight":
                # Check if it's a slight movement or a significant one
                if head_code.startswith("slight_"):
                    self.head_tracking_stats["slight_movement_count"] += 1
                else:
                    self.head_tracking_stats["head_movement_count"] += 1
                
                self.head_tracking_stats["last_movement_time"] = current_time
            
            # Also count transitions between different slight movements
            elif prev_head_code.startswith("slight_") and head_code.startswith("slight_") and prev_head_code != head_code:
                self.head_tracking_stats["slight_movement_count"] += 1
                self.head_tracking_stats["last_movement_time"] = current_time
            
            # Update the direction and vector
            self.head_tracking_stats["head_direction"] = head_code
            self.head_tracking_stats["head_direction_vector"] = direction_vector
            self.head_tracking_stats["last_status_change_time"] = current_time
        else:
            # Even if the code hasn't changed, update the direction vector for continuous tracking
            self.head_tracking_stats["head_direction_vector"] = direction_vector
        
        # If the head is straight, add to straight head time
        if head_code == "head_straight":
            self.head_tracking_stats["straight_head_time"] += elapsed
        
        # Calculate straight time percentage
        total_tracking_time = current_time - self.head_tracking_stats["tracking_start_time"]
        if total_tracking_time > 0:
            self.head_tracking_stats["straight_time_percentage"] = (
                self.head_tracking_stats["straight_head_time"] / total_tracking_time * 100
            )
        
        return head_status
    
    def process_frame(self, frame):
        """Process a single frame for head detection"""
        if frame is None:
            return frame  # Return original frame if None
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face landmarks
        results = self.face_mesh.process(rgb_frame)
        head_status = "No Face Detected"
        head_code = "no_face"
        
        # Create a clean frame - just the original frame without modifications
        output_frame = frame.copy()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Estimate head pose
                roll, pitch, yaw, direction_vector = self.estimate_head_pose(face_landmarks.landmark, w, h)
                
                # Apply smoothing
                roll, pitch, yaw, direction_vector = self.smooth_angles(roll, pitch, yaw, direction_vector)
                
                # During calibration, collect data for reference position
                if not self.calibration_complete:
                    self.frame_count += 1
                    self.calibration_data.append((roll, pitch, yaw))
                    
                    if self.frame_count >= self.CALIBRATION_FRAMES:
                        # Calculate reference angles and thresholds
                        ref_roll, ref_pitch, ref_yaw = self.calibrate_head_pose(self.calibration_data)
                        if ref_roll is not None:
                            self.reference_roll, self.reference_pitch, self.reference_yaw = ref_roll, ref_pitch, ref_yaw
                            self.calibration_complete = True
                        else:
                            # Reset calibration if it failed
                            self.frame_count = 0
                            self.calibration_data = []
                else:
                    # Classify head direction with improved sensitivity
                    head_code, head_status, direction_vector = self.classify_head_direction(
                        roll, pitch, yaw, direction_vector)
                    
                    # Update the head tracking metrics
                    head_status = self.update_head_metrics(head_code, head_status, direction_vector)
        
        # Store current head status
        self.current_head_status = head_status
        
        # Return the original frame for display
        return frame
    
    def get_output_frame(self):
        """Get the current frame without any modifications"""
        return None  # Don't return a modified frame
    
    def get_current_head_status(self):
        """Get the current head status for external use"""
        return self.current_head_status
    
    def get_head_tracking_metrics(self):
        """Get head tracking metrics for external use"""
        return self.head_tracking_stats