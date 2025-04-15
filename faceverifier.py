import cv2
import numpy as np
import os
import sqlite3
import face_recognition
import logging
from datetime import datetime

class FaceVerifier:
    def __init__(self, db_path="face_database.db", photos_dir="passport_photos"):
        """Initialize the face verifier with database and photo storage directory"""
        # Set up logging
        self.logger = logging.getLogger("face_verifier")
        
        # Database for storing face encodings and user information
        self.db_path = db_path
        self._setup_database()
        
        # Directory for storing passport photos
        self.photos_dir = photos_dir
        if not os.path.exists(photos_dir):
            os.makedirs(photos_dir)
        
        # Face recognition parameters
        self.face_encoding_model = "hog"  # More efficient than CNN for real-time processing
        self.face_match_tolerance = 0.6   # Lower is more strict matching
        
        # Status tracking
        self.last_verification_time = None
        self.last_verification_result = None
        self.verification_count = 0
        self.successful_verifications = 0

    def _setup_database(self):
        """Set up the SQLite database for face verification"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table for storing user info
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                photo_path TEXT,
                registration_date TEXT,
                last_verification TEXT
            )
            ''')
            
            # Create face_encodings table for storing face feature vectors
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                encoding BLOB,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Create verification_logs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_logs (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                timestamp TEXT,
                result TEXT,
                confidence REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            
    def register_user(self, user_name, image):
        """
        Register a new user with their passport photo
        
        Args:
            user_name (str): Name of the user
            image (numpy.ndarray): Passport photo image
            
        Returns:
            tuple: (success (bool), user_id or error message (int or str))
        """
        try:
            # Get the best face encoding from the passport photo
            face_encoding = self.get_best_face_encoding(image)
            
            if face_encoding is None:
                return False, "No face detected in the passport photo"
            
            # Save the passport photo
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            photo_filename = f"{user_name.replace(' ', '_')}_{timestamp}.jpg"
            photo_path = os.path.join(self.photos_dir, photo_filename)
            cv2.imwrite(photo_path, image)
            
            # Save user details and face encoding to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert user details
            cursor.execute(
                "INSERT INTO users (name, photo_path, registration_date, last_verification) VALUES (?, ?, ?, ?)",
                (user_name, photo_path, datetime.now().isoformat(), None)
            )
            user_id = cursor.lastrowid
            
            # Insert face encoding
            encoding_blob = face_encoding.tobytes()
            cursor.execute(
                "INSERT INTO face_encodings (user_id, encoding) VALUES (?, ?)",
                (user_id, encoding_blob)
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"User {user_name} registered successfully with ID {user_id}")
            return True, user_id
            
        except Exception as e:
            self.logger.error(f"Error registering user: {str(e)}")
            return False, str(e)
    
    def verify_face(self, frame, user_id=None):
        """
        Verify if the face in the frame matches the registered user
        
        Args:
            frame (numpy.ndarray): Current video frame
            user_id (int, optional): Specific user ID to verify against
                                     If None, verify against all users
                                     
        Returns:
            tuple: (match_found (bool), user_id or message (int or str), confidence (float))
        """
        try:
            # Get the best face encoding from the current frame
            current_encoding = self.get_best_face_encoding(frame)
            
            if current_encoding is None:
                return False, "No face detected in frame", 0.0
            
            # Connect to database to retrieve stored encodings
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query the database based on whether a specific user is requested
            if user_id is not None:
                cursor.execute(
                    "SELECT users.id, users.name, face_encodings.encoding FROM users JOIN face_encodings ON users.id = face_encodings.user_id WHERE users.id = ?", 
                    (user_id,)
                )
            else:
                cursor.execute(
                    "SELECT users.id, users.name, face_encodings.encoding FROM users JOIN face_encodings ON users.id = face_encodings.user_id"
                )
            
            results = cursor.fetchall()
            
            if not results:
                conn.close()
                return False, "No registered users found", 0.0
            
            # Compare the current face encoding with stored encodings
            best_match = None
            best_match_distance = float('inf')
            
            for db_user_id, name, encoding_blob in results:
                # Convert blob back to numpy array
                stored_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                
                # Calculate face distance (lower is more similar)
                face_distance = face_recognition.face_distance([stored_encoding], current_encoding)[0]
                
                # Keep track of the best match
                if face_distance < best_match_distance:
                    best_match_distance = face_distance
                    best_match = (db_user_id, name, face_distance)
            
            # Determine if this is a match based on the tolerance threshold
            # Apply a more flexible tolerance for verification
            adaptive_tolerance = self.face_match_tolerance * 1.1  # 10% more lenient
            
            if best_match is not None:
                match_id, match_name, distance = best_match
                # Convert distance to confidence (0-1 scale, higher is better)
                confidence = 1 - min(distance, 1.0)
                is_match = distance <= adaptive_tolerance
                
                # Update verification statistics
                self.verification_count += 1
                if is_match:
                    self.successful_verifications += 1
                self.last_verification_time = datetime.now()
                self.last_verification_result = is_match
                
                # Log the verification
                cursor.execute(
                    "INSERT INTO verification_logs (user_id, timestamp, result, confidence) VALUES (?, ?, ?, ?)",
                    (match_id, self.last_verification_time.isoformat(), "success" if is_match else "failure", confidence)
                )
                
                # Update the user's last verification timestamp
                if is_match:
                    cursor.execute(
                        "UPDATE users SET last_verification = ? WHERE id = ?",
                        (self.last_verification_time.isoformat(), match_id)
                    )
                
                conn.commit()
                conn.close()
                
                if is_match:
                    return True, match_id, confidence
                else:
                    return False, f"No match found (closest: {match_name} with {confidence*100:.1f}% confidence)", confidence
            
            conn.close()
            return False, "No match found in database", 0.0
            
        except Exception as e:
            self.logger.error(f"Error during face verification: {str(e)}")
            return False, str(e), 0.0
    
    def get_user_photo_path(self, user_id):
        """Get the passport photo path for a specific user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT photo_path FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0]
            else:
                return None
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
            "last_verification_result": self.last_verification_result
        }

    def get_all_users(self):
        """Get a list of all registered users"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, photo_path, registration_date, last_verification FROM users")
            users = [
                {
                    "id": row[0],
                    "name": row[1],
                    "photo_path": row[2],
                    "registration_date": row[3],
                    "last_verification": row[4]
                }
                for row in cursor.fetchall()
            ]
            conn.close()
            return users
        except Exception as e:
            self.logger.error(f"Error retrieving users: {str(e)}")
            return []

    def delete_user(self, user_id):
        """Delete a user and their face encodings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get photo path first
            cursor.execute("SELECT photo_path FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                photo_path = result[0]
                # Delete the photo file if it exists
                if os.path.exists(photo_path):
                    os.remove(photo_path)
            
            # Delete related face encodings
            cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
            
            # Delete verification logs
            cursor.execute("DELETE FROM verification_logs WHERE user_id = ?", (user_id,))
            
            # Delete the user
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Error deleting user: {str(e)}")
            return False
        
    def get_best_face_encoding(self, image, num_samples=3):
        """
        Get the best face encoding from multiple samples of the same face
        to improve accuracy
        
        Args:
            image (numpy.ndarray): Image containing a face
            num_samples (int): Number of slightly different encodings to generate
            
        Returns:
            numpy.ndarray: The best face encoding or None if no face detected
        """
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
        main_face_location = [face_locations[largest_face_idx]]
        
        # Create slightly modified versions of the main face location
        # This helps with slight variations in face position
        modified_face_locations = []
        modified_face_locations.append(main_face_location[0])  # Add the original face
        
        top, right, bottom, left = main_face_location[0]
        # Add slightly expanded face
        expanded = (top - 2, right + 2, bottom + 2, left - 2)
        modified_face_locations.append(expanded)
        # Add slightly contracted face
        contracted = (top + 2, right - 2, bottom - 2, left + 2)
        modified_face_locations.append(contracted)
        
        # Get encodings for all variations
        encodings = []
        for loc in modified_face_locations[:num_samples]:  # Limit to num_samples
            try:
                face_enc = face_recognition.face_encodings(image, [loc])[0]
                encodings.append(face_enc)
            except IndexError:
                # Skip if encoding fails for a modified location
                continue
        
        if not encodings:
            return None
            
        # Return the first valid encoding (most reliable)
        return encodings[0]