Drowsiness Detection Using OpenCV and dlib
This repository provides an implementation of a Drowsiness Detection System that utilizes OpenCV, dlib, and Eye Aspect Ratio (EAR) to monitor a person's eye movement and detect drowsiness. The system processes video frames to determine whether the person is drowsy based on prolonged eye closure, which is a common sign of fatigue.

Table of Contents
Installation
Usage
Requirements
Credits
Installation
1. Clone the repository
bash
Copy
git clone(https://github.com/lasyaistla/Drowsiness-detection)
cd drowsiness-detection
2. Install the necessary dependencies
Run the following command to install the required packages:

bash
Copy
pip install -r requirements.txt
If you're using Google Colab, you can install the dependencies by running:

python
Copy
!pip install opencv-python opencv-python-headless dlib imutils
3. Download the dlib face landmark model
To use facial landmark detection, download the shape_predictor_68_face_landmarks.dat model from here and extract it into your project directory.

4. Upload Your Video
If you're running the code on Google Colab, upload the video file directly by using:

python
Copy
from google.colab import files
uploaded = files.upload()
Then, make sure to use the correct video path in your script.

Usage
1. Set up the paths
Set the paths for the shape_predictor_68_face_landmarks.dat and your video file.

python
Copy
shape_predictor_path = '/content/shape_predictor_68_face_landmarks.dat'  # Replace with actual path
video_path = '/content/your_video.mp4'  # Replace with the path of your uploaded video file
2. Run the detection script
The script will process the video frame by frame, detect faces, compute the Eye Aspect Ratio (EAR), and identify drowsiness based on the threshold value of EAR.

3. Start processing
Run the following Python code to detect drowsiness in the video:

python
Copy
import cv2
import dlib
import numpy as np
from imutils import face_utils
from google.colab.patches import cv2_imshow

# Load the face detector and face landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Load the video
cap = cv2.VideoCapture(video_path)

# Set threshold and frame count for drowsiness detection
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 48
frame_count = 0
drowsy = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        # Check if the EAR is below the threshold (indicating drowsiness)
        if ear < EAR_THRESHOLD:
            frame_count += 1
            if frame_count >= CONSECUTIVE_FRAMES:
                drowsy = True
        else:
            frame_count = 0
            drowsy = False

        cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display a drowsiness alert if detected
    if drowsy:
        cv2.putText(frame, "DROWSINESS ALERT!", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Show the processed frame
    cv2_imshow(frame)

# Release the video capture object
cap.release()
4. Observe the Output
The system will display the video frames with the Eye Aspect Ratio (EAR) value on the screen. If drowsiness is detected, a DROWSINESS ALERT will be shown on the frame.
#Adding few output picture i used 1min long video as an input.

![Screenshot 2025-01-31 184418](https://github.com/user-attachments/assets/83198eb9-dc09-4976-afe9-4263272e8521)
![Screenshot 2025-01-31 184505](https://github.com/user-attachments/assets/3baf7d3d-1352-41de-941a-79aa1ead47c4)
![Screenshot 2025-01-31 184525](https://github.com/user-attachments/assets/4d9d7f80-a0f0-4fec-8808-cd35322af9f8)

Requirements
Python 3.x
OpenCV: For video processing and face detection.
dlib: For facial landmark detection.
imutils: For simplified image processing.
Google Colab (optional): For running the code in the cloud without requiring local resources.
Credits
OpenCV: https://opencv.org/
dlib: http://dlib.net/
Shape Predictor Model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Google Colab: https://colab.research.google.com/
