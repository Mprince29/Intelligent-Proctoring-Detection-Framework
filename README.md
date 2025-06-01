🎓 Intelligent-Proctoring-Detection-Framework

A real-time AI powered invigilation system built with Flask, OpenCV, and multiple detection modules. This tool monitors students during online exams and detects suspicious behavior such as looking away, unauthorized objects, multiple faces, and speaking.



🚀 Features

	•	🔍 Eye Tracker — Detects if the user’s gaze deviates from the screen.
	•	🙆 Head Detector — Monitors head movement direction.
	•	🧑‍🤝‍🧑 Face Verifier — Checks for unauthorized additional faces.
	•	🎤 Audio Detector — Flags background or user speech.
	•	📦 Object Detector — Detects non-allowed physical objects.
	•	📷 Webcam Integration — Captures live feed using OpenCV.
	•	📝 PDF Report Generator — Creates session summary in downloadable format.
	•	☁️ MongoDB Logging — Stores session logs and evidence.
	•	🔐 Secure Upload — Upload and store verification media safely.
	•	🖥️ Live Surveillance View — Continuous camera streaming.



🧠 Architecture Overview

Frontend (HTML + Flask templates)
        
Flask Server (app.py)
        
        +--> EyeTracker (eye_tracker.py)
        
        +--> HeadDetector (head_detector.py)
        
        +--> FaceVerifier (faceverifier.py)
        
        +--> AudioDetector (audio_detector.py)
        
        +--> ObjectDetector (object_detector.py)
        
        
MongoDB (Logging events, timestamps, violations)


📁 your_project/

├── app.py                    # Main Flask application

├── eye_tracker.py           # Eye tracking logic

├── head_detector.py         # Head movement detection

├── faceverifier.py          # Face verification logic

├── audio_detector.py        # Speech detection

├── object_detector.py       # Object detection model

├── templates/               # HTML templates

├── static/                  # CSS, JS, images

├── uploads/                 # Uploaded reference images/videos

└── requirements.txt         # Python dependencies


⚙️ How It Works

	1.	Launch Flask App:

 python app.py

 Visit http://localhost:5001 in your browser.

	2.	Upload User Data:
	•	Provide student name, ID, and a reference face image before starting the exam.
	3.	Start Surveillance:
	•	System captures webcam feed.
	•	Runs detection models in real-time (on separate threads).
	•	Logs events (e.g., “Multiple faces detected”, “Looking away”, “Talking”).
	4.	Report Generation:
	•	After session ends, a PDF report can be downloaded.
	•	Includes timestamps, violation types, and optionally screenshots.


🧪 Example Use Case

	1.	Student logs in → Uploads ID & face image.
 
	2.	Invigilator starts session → Models monitor for cheating.
 
	3.	Student looks away/talks → Event is logged.
 
	4.	Session ends → Invigilator downloads detailed report.


📦 Installation

# Clone repository

git clone https://github.com/Mprince29/Intelligent-Proctoring-Detection-Framework.git

cd Intelligent-Proctoring-Detection-Framework

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py

📋 Dependencies
	•	Flask
	•	OpenCV (cv2)
	•	reportlab
	•	numpy
	•	pymongo
	•	werkzeug
	•	Your custom modules (eye_tracker.py, etc.)

🛡️ Security Notes
	•	Ensure secure webcam access via HTTPS in production.
	•	Avoid hardcoding secrets. Use environment variables instead.
	•	Validate uploaded files to prevent injection attacks.

🧰 Tips for Developers
	•	Models are instantiated globally for performance.
	•	Threading is used for real-time parallel analysis.
	•	MongoDB stores all session logs and metadata.
	•	All user-uploaded files are stored in the uploads/ folder.

📄 Example Report Output
	•	Timestamped list of all detected violations.
	•	Session duration and student metadata.
	•	Optional snapshots captured at time of violation.
