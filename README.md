**🛡️ Cheating Detection System (ML-Powered Proctoring)**

An advanced ML-based proctoring system built using Flask and OpenCV, incorporating real-time eye tracking, head movement detection,
face verification, object detection, and audio monitoring to ensure cheating-free environments during online exams or assessments.

📌 Features

🔍 Eye Tracking — Detects gaze direction and reading behavior.

🧠 Head Movement Detection — Monitors head orientation to detect distractions.

🧑‍💻 Face Verification — Confirms and verifies student identity using face recognition.

🎧 Audio Monitoring — Detects voice activity, background conversations, and checks for headphone usage.

📦 Object Detection — Detects unauthorized objects or devices in the environment.

📝 Report Generation — Generates a detailed PDF report of all monitoring metrics.

🧾 User Registration — Register users with photos for identity verification.

🔄 Live Video Feed — Shows a real-time annotated camera stream.

📊 JSON API — Provides real-time data for eye/head/audio/object metrics.

⚙️ Tech Stack

Backend: Python, Flask, OpenCV, NumPy

Face & Eye Tracking: Custom ML modules (EyeTracker, HeadDetector, FaceVerifier)

Audio Analysis: PyAudio, Speech Recognition

Object Detection: OpenCV DNN or custom model

PDF Report Generation: ReportLab

Frontend: HTML (via Flask templates)

🧪 Core Modules

Module	Purpose

EyeTracker	Tracks eye movement, blinking, and gaze
HeadDetector	Detects head orientation and movement
FaceVerifier	Matches face with registered photo
AudioDetector	Monitors speech, headphones, audio levels
ObjectDetector	Detects unauthorized objects in camera feed

📁 Folder Structure

├── app.py

├── eye_tracker.py

├── head_detector.py

├── audio_detector.py

├── object_detector.py

├── faceverifier.py

├── passport_photos/

├── templates/

│   ├── index.html

│   └── register.html

├── static/

├── face_database.db

🧑‍💻 Author

Master Prince


