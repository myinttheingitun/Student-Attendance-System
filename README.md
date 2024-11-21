# Face Detection Attendance System

This project utilizes Python and OpenCV using PyCharm to create an automated student attendance system based on face recognition. It uses a webcam to recognize faces and match them with a pre-recorded dataset.

## Features
- **Data Registration**: Students provide their name and ID to create a training dataset.
- **Face Recognition**: Real-time attendance tracking via face recognition.
- **Voice Confirmation**: After a successful match, Press 'O' for attendence taken and a voice prompt says "Attendance taken" .
- **Quick Exit**: Press 'Q' to quickly exit the system.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnnieWillow/FaceDetection.git

## Usage
1. Training Mode: Run Dataset.py to register student data by entering their name and ID.
   
   * The system trains the model using the provided data.


    ```bash
   python Dataset.py
   

2. Attendance Mode: Run AttendanceRecord.py to track attendance with face recognition.
   
   * Press 'O' to record attendance when the system recognizes a student.
   * Press 'Q' to exit the system.

   
   ```bash
   python Attendance.py
   
   
## Files

- Dataset.py: Script to collect training data (name, ID, and face images).
- Attendance.py: Runs the real-time face recognition for attendance.
- haarcascade_frontalface_default.xml: Pre-trained face detection model used by OpenCV for webcam processing.
- attendance.csv: Stores attendance records with up-to-date time and date format.
- CV project.png: Background image used in the application interface.

## Contributing
Feel free to fork and submit pull requests for improvements.

