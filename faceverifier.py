import cv2
import numpy as np
import os
import logging
import face_recognition
from datetime import datetime
from PIL import Image
from pymongo import MongoClient
from bson import Binary, ObjectId
import dlib
import time

class FaceVerifier:
    def __init__(self, db_uri="mongodb://localhost:27017/", db_name="face_verification", photos_dir="passport_photos"):
        """Initialize the face verifier with database and photo storage directory"""
        # Set up logging
        self.logger = logging.getLogger("face_verifier")
        self.logger.setLevel(logging.DEBUG)
        
        # Connect to MongoDB
        try:
            self.client = MongoClient(db_uri)
            self.db = self.client[db_name]
            self.db.users.create_index("name")
            self.logger.info("MongoDB connection established successfully")
        except Exception as e:
            self.logger.error(f"MongoDB connection error: {str(e)}")
            raise
        
        # Directory for storing passport photos
        self.photos_dir = photos_dir
        if not os.path.exists(photos_dir):
            os.makedirs(photos_dir)
        
        # Face recognition parameters - Focus on live verification
        self.face_encoding_model = "hog"
        self.num_encoding_samples = 5      # Number of samples to take from photo
        self.live_samples_required = 10    # Number of samples required for live verification
        self.consecutive_matches_required = 5  # Number of consecutive matches required
        self.verification_window = 30      # Time window for verification in seconds
        self.min_confidence_threshold = 0.5  # Minimum confidence for a match
        
        # Initialize face detector
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Verification tracking
        self.verification_buffer = []  # Store recent verification attempts
        self.last_verification_time = None
        self.verification_start_time = None
        self.consecutive_matches = 0
        self.last_matched_id = None
        self.verification_count = 0
        self.successful_verifications = 0

    def assess_image_quality(self, image):
        """
        Assess the quality of the face image
        Returns: quality_score (float), message (str)
        """
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Check image resolution
            height, width = gray.shape
            if width < 250 or height < 250:
                return 0.0, "Image resolution too low"

            # Check brightness
            brightness = np.mean(gray)
            if brightness < 40 or brightness > 250:
                return 0.0, "Poor lighting conditions"

            # Check contrast
            contrast = np.std(gray)
            if contrast < 20:
                return 0.0, "Image contrast too low"

            # Detect face and facial landmarks
            faces = self.face_detector(gray)
            if len(faces) != 1:
                return 0.0, "Multiple faces detected or no face detected"

            face = faces[0]
            face_width = face.right() - face.left()
            face_height = face.bottom() - face.top()

            # Check face size relative to image
            face_area_ratio = (face_width * face_height) / (width * height)
            if face_area_ratio < 0.1:
                return 0.0, "Face too small in image"
            if face_area_ratio > 0.9:
                return 0.0, "Face too close to camera"

            # Calculate quality score
            quality_score = min(1.0, (contrast / 100) * (face_area_ratio * 2))
            return quality_score, "Acceptable image quality"

        except Exception as e:
            self.logger.error(f"Error assessing image quality: {str(e)}")
            return 0.0, "Error assessing image quality"

    def register_user(self, name, email, phone, photo_path):
        """Register a new user with their face encoding"""
        try:
            # Load and preprocess the registration photo
            image = face_recognition.load_image_file(photo_path)
            image = self.preprocess_image(image)
            
            # Get face encodings from the photo
            face_encodings = []
            for _ in range(self.num_encoding_samples):
                encoding = self.get_best_face_encoding(image)
                if encoding is not None:
                    face_encodings.append(encoding)
            
            if not face_encodings:
                self.logger.error(f"No face detected in photo for user {name}")
                return None
            
            # Use the average encoding
            average_encoding = np.mean(face_encodings, axis=0)
            average_encoding = average_encoding / np.linalg.norm(average_encoding)
            
            # Create user document
            user_doc = {
                "name": name,
                "email": email,
                "phone": phone,
                "photo_path": photo_path,
                "face_encoding": Binary(average_encoding.tobytes()),
                "registration_date": datetime.now(),
                "last_verification": None,
                "verification_logs": []
            }
            
            # Insert the user document
            result = self.db.users.insert_one(user_doc)
            
            if result.inserted_id:
                self.logger.info(f"Successfully registered user {name}")
                return str(result.inserted_id)
            else:
                self.logger.error(f"Failed to insert user document for {name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error registering user: {str(e)}")
            return None

    def verify_face(self, frame, user_id=None):
        """Verify if the face in the frame matches the registered user"""
        try:
            current_time = time.time()
            
            # Initialize verification session if needed
            if user_id != self.last_matched_id:
                self.reset_verification_session(user_id)
            elif self.verification_start_time is None:
                self.verification_start_time = current_time
            
            # Check if we're still within the verification window
            if self.verification_start_time and (current_time - self.verification_start_time) > self.verification_window:
                self.reset_verification_session(user_id)
                return False, "Verification timeout - please try again", 0.0
            
            # Process the frame
            frame = self.preprocess_image(frame)
            current_encoding = self.get_best_face_encoding(frame)
            
            if current_encoding is None:
                self.consecutive_matches = 0
                return False, "No face detected", 0.0
            
            # Query user(s)
            query = {"_id": ObjectId(user_id)} if user_id else {}
            users = list(self.db.users.find(query))
            
            if not users:
                return False, "No registered users found", 0.0
            
            # Compare with each user
            best_match = None
            best_match_confidence = 0.0
            
            for user in users:
                stored_encoding = np.frombuffer(user['face_encoding'], dtype=np.float64)
                face_distance = face_recognition.face_distance([stored_encoding], current_encoding)[0]
                confidence = 1 - face_distance
                
                if confidence > best_match_confidence:
                    best_match = (str(user['_id']), user['name'], confidence)
                    best_match_confidence = confidence
            
            if best_match and best_match_confidence >= self.min_confidence_threshold:
                # Store verification attempt
                self.verification_buffer.append({
                    'time': current_time,
                    'confidence': best_match_confidence,
                    'user_id': best_match[0]
                })
                
                # Keep only recent attempts
                self.verification_buffer = [
                    v for v in self.verification_buffer 
                    if (current_time - v['time']) <= self.verification_window
                ]
                
                # Check verification criteria
                if len(self.verification_buffer) >= self.live_samples_required:
                    # Calculate average confidence
                    recent_confidences = [v['confidence'] for v in self.verification_buffer[-self.live_samples_required:]]
                    avg_confidence = np.mean(recent_confidences)
                    
                    if avg_confidence >= self.min_confidence_threshold:
                        self.consecutive_matches += 1
                        if self.consecutive_matches >= self.consecutive_matches_required:
                            # Successful verification
                            self._log_verification_attempt(best_match[0], True, avg_confidence, "live")
                            return True, best_match[0], avg_confidence * 100
                    else:
                        self.consecutive_matches = 0
                
                # Still collecting samples
                return False, f"Verifying ({len(self.verification_buffer)}/{self.live_samples_required} samples)", best_match_confidence * 100
            
            # No match found
            self.consecutive_matches = 0
            return False, "No match found", 0.0
            
        except Exception as e:
            self.logger.error(f"Error during face verification: {str(e)}")
            return False, str(e), 0.0

    def reset_verification_session(self, new_user_id=None):
        """Reset the verification session"""
        self.verification_buffer = []
        self.verification_start_time = None
        self.consecutive_matches = 0
        self.last_matched_id = new_user_id

    def _log_verification_attempt(self, user_id, is_match, confidence, verification_type):
        """Log a verification attempt"""
        try:
            log_entry = {
                "timestamp": datetime.now(),
                "result": "success" if is_match else "failure",
                "confidence": confidence,
                "verification_type": verification_type
            }
            
            update = {
                "$push": {"verification_logs": log_entry}
            }
            
            if is_match:
                update["$set"] = {"last_verification": log_entry["timestamp"]}
            
            self.db.users.update_one(
                {"_id": ObjectId(user_id)},
                update
            )
            
            self.verification_count += 1
            if is_match:
                self.successful_verifications += 1
            self.last_verification_time = log_entry["timestamp"]
            
        except Exception as e:
            self.logger.error(f"Error logging verification attempt: {str(e)}")

    def verify_face_symmetry(self, frame):
        """Verify face symmetry as an additional security check"""
        try:
            face_landmarks = face_recognition.face_landmarks(frame)
            if not face_landmarks:
                return False
            
            landmarks = face_landmarks[0]
            
            # Check eye symmetry
            left_eye = np.mean(landmarks['left_eye'], axis=0)
            right_eye = np.mean(landmarks['right_eye'], axis=0)
            eye_distance = np.linalg.norm(left_eye - right_eye)
            
            # Check mouth symmetry
            top_lip = np.mean(landmarks['top_lip'], axis=0)
            bottom_lip = np.mean(landmarks['bottom_lip'], axis=0)
            mouth_height = np.linalg.norm(top_lip - bottom_lip)
            
            # Calculate symmetry ratio
            symmetry_ratio = mouth_height / eye_distance
            
            # Check if the ratio is within normal range (typically 0.3-0.5)
            return 0.3 <= symmetry_ratio <= 0.5
            
        except Exception as e:
            self.logger.error(f"Error checking face symmetry: {str(e)}")
            return False

    def _serialize_landmarks(self, landmarks):
        """Convert landmarks dictionary to serializable format"""
        return {k: [list(map(float, point)) for point in v] for k, v in landmarks.items()}

    def verify_facial_landmarks(self, frame, user_id):
        """
        Additional verification using facial landmarks
        """
        try:
            # Get facial landmarks from the current frame
            face_locations = face_recognition.face_locations(frame, model=self.face_encoding_model)
            if not face_locations:
                return False
                
            current_landmarks = face_recognition.face_landmarks(frame, face_locations)
            if not current_landmarks:
                return False
            
            # Get the stored photo for comparison
            user = self.db.users.find_one({"_id": ObjectId(user_id)})
            if not user or not user['photo_path']:
                return False
                
            stored_image = face_recognition.load_image_file(user['photo_path'])
            stored_locations = face_recognition.face_locations(stored_image, model=self.face_encoding_model)
            if not stored_locations:
                return False
                
            stored_landmarks = face_recognition.face_landmarks(stored_image, stored_locations)
            if not stored_landmarks:
                return False
            
            # Compare key facial features
            current_features = current_landmarks[0]
            stored_features = stored_landmarks[0]
            
            # Check relative positions of key features
            features_match = self._compare_facial_features(current_features, stored_features)
            
            return features_match
            
        except Exception as e:
            self.logger.error(f"Error in landmark verification: {str(e)}")
            return False

    def _compare_facial_features(self, current_features, stored_features):
        """
        Compare relative positions of facial features
        """
        try:
            # Calculate center points of key features
            def get_center_point(points):
                return np.mean(points, axis=0)
            
            # Compare key features
            features_to_check = ['nose_bridge', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
            reference_points = {}
            
            for feature in features_to_check:
                if feature in current_features and feature in stored_features:
                    current_center = get_center_point(current_features[feature])
                    stored_center = get_center_point(stored_features[feature])
                    reference_points[feature] = (current_center, stored_center)
            
            # Calculate relative distances between features
            matches = 0
            total_checks = 0
            
            for i, feature1 in enumerate(features_to_check):
                for feature2 in features_to_check[i+1:]:
                    if feature1 in reference_points and feature2 in reference_points:
                        current_dist = np.linalg.norm(
                            reference_points[feature1][0] - reference_points[feature2][0])
                        stored_dist = np.linalg.norm(
                            reference_points[feature1][1] - reference_points[feature2][1])
                        
                        # Compare relative distances with tolerance
                        if abs(current_dist - stored_dist) / stored_dist < 0.15:  # 15% tolerance
                            matches += 1
                        total_checks += 1
            
            # Require at least 70% of feature relationships to match
            return total_checks > 0 and (matches / total_checks) >= 0.7
            
        except Exception as e:
            self.logger.error(f"Error comparing facial features: {str(e)}")
            return False

    def get_user_photo_path(self, user_id):
        """Get the passport photo path for a specific user"""
        try:
            user = self.db.users.find_one({"_id": ObjectId(user_id)})
            return user['photo_path'] if user else None
        except Exception as e:
            self.logger.error(f"Error retrieving user photo path: {str(e)}")
            return None

    def get_verification_metrics(self):
        """Get verification performance metrics"""
        success_rate = 0
        if self.verification_count > 0:
            success_rate = (self.successful_verifications / self.verification_count) * 100
            
        return {
            "verification_count": self.verification_count,
            "successful_verifications": self.successful_verifications,
            "success_rate": success_rate,
            "last_verification_time": self.last_verification_time.isoformat() if self.last_verification_time else None,
        }

    def get_all_users(self):
        """Get a list of all registered users"""
        try:
            users = []
            for user in self.db.users.find({}, {
                "name": 1,
                "photo_path": 1,
                "registration_date": 1,
                "last_verification": 1
            }):
                users.append({
                    "id": str(user['_id']),
                    "name": user.get('name', 'Unknown'),
                    "photo_path": user.get('photo_path', ''),
                    "registration_date": user.get('registration_date', datetime.now()).isoformat(),
                    "last_verification": user.get('last_verification', None).isoformat() if user.get('last_verification') else None
                })
            return users
        except Exception as e:
            self.logger.error(f"Error retrieving users: {str(e)}")
            return []

    def delete_user(self, user_id):
        """Delete a user and their face encodings"""
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(user_id)
            
            # Get user document first
            user = self.db.users.find_one({"_id": object_id})
            
            if user and user['photo_path']:
                # Delete the photo file if it exists
                if os.path.exists(user['photo_path']):
                    os.remove(user['photo_path'])
            
            # Delete the user document (includes embedded face encoding and logs)
            result = self.db.users.delete_one({"_id": object_id})
            return result.deleted_count > 0
            
        except Exception as e:
            self.logger.error(f"Error deleting user: {str(e)}")
            return False

    def preprocess_image(self, image):
        """
        Preprocess image for better face detection and recognition
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize if image is too large
            max_size = 1024
            height, width = image.shape[:2]
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))

            # Enhance image
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert to PIL Image for face_recognition library
            image = Image.fromarray(image)
            
            return np.array(image)
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return image

    def align_face(self, image, face_location):
        """
        Align face based on eye positions for better recognition
        """
        try:
            # Get facial landmarks
            face_landmarks = face_recognition.face_landmarks(image, [face_location])
            if not face_landmarks:
                return image

            landmarks = face_landmarks[0]
            
            # Get eye coordinates
            left_eye = np.mean(landmarks['left_eye'], axis=0)
            right_eye = np.mean(landmarks['right_eye'], axis=0)
            
            # Calculate angle to align eyes horizontally
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Get the center of the face
            center = (int((face_location[1] + face_location[3]) / 2),
                     int((face_location[0] + face_location[2]) / 2))
            
            # Create rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Perform the rotation
            aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                         flags=cv2.INTER_CUBIC)
            
            return aligned_image
        except Exception as e:
            self.logger.error(f"Error aligning face: {str(e)}")
            return image

    def get_best_face_encoding(self, image, num_samples=5):
        """
        Get the best face encoding from multiple samples of the same face
        to improve accuracy
        
        Args:
            image (numpy.ndarray): Image containing a face
            num_samples (int): Number of slightly different encodings to generate
            
        Returns:
            numpy.ndarray: The best face encoding or None if no face detected
        """
        # Preprocess the image
        image = self.preprocess_image(image)
        
        # First detect all faces in the image
        face_locations = face_recognition.face_locations(image, model=self.face_encoding_model)
        
        if not face_locations:
            return None
        
        # Find the largest face (presumably the main face)
        largest_face_idx = 0
        largest_face_size = 0
        
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_size = (bottom - top) * (right - left)
            if face_size > largest_face_size:
                largest_face_size = face_size
                largest_face_idx = i
        
        # Get the main face location
        main_face_location = face_locations[largest_face_idx]
        
        # Align the face
        aligned_image = self.align_face(image, main_face_location)
        
        # Create slightly modified versions of the main face location
        modified_face_locations = []
        top, right, bottom, left = main_face_location
        
        # Add original face
        modified_face_locations.append((top, right, bottom, left))
        
        # Add variations with different crops and scales
        variations = [
            (top - 2, right + 2, bottom + 2, left - 2),  # Expanded
            (top + 2, right - 2, bottom - 2, left + 2),  # Contracted
            (top - 1, right + 1, bottom + 1, left - 1),  # Slightly expanded
            (top + 1, right - 1, bottom - 1, left + 1),  # Slightly contracted
        ]
        
        modified_face_locations.extend(variations)
        
        # Get encodings for all variations
        encodings = []
        for loc in modified_face_locations[:num_samples]:
            try:
                face_enc = face_recognition.face_encodings(aligned_image, [loc])[0]
                encodings.append(face_enc)
            except IndexError:
                continue
        
        if not encodings:
            return None
        
        # Calculate the average encoding
        average_encoding = np.mean(encodings, axis=0)
        # Normalize the average encoding
        average_encoding = average_encoding / np.linalg.norm(average_encoding)
        
        return average_encoding