USE  Detection;
-- Users table to store basic user information
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verification TIMESTAMP,
    photo_path TEXT NOT NULL
);

-- Face encodings table to store facial recognition data
CREATE TABLE face_encodings (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    encoding BLOB NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Verification logs to track all verification attempts
CREATE TABLE verification_logs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result TEXT NOT NULL,  -- 'success' or 'failure'
    confidence REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- Eye tracking sessions to link verification with tracking data
CREATE TABLE eye_tracking_sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    verification_status TEXT NOT NULL,  -- 'verified', 'failed', 'incomplete'
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);