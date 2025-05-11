# Face Mask Detection System
The Face Mask Detection System is a machine learning-based web application designed to detect whether individuals are wearing face masks. This application can be used in public spaces, offices, hospitals, or any organization to enhance safety protocols and prevent the spread of airborne diseases. The system uses computer vision techniques to identify faces and classify them based on mask-wearing status.
# Objective
The primary objective of this project is to automate the process of mask detection using real-time image processing. This helps ensure adherence to safety protocols with minimal manual supervision.
# Key Features
**Image-Based Detection**
* Upload an image to detect and classify faces as "Mask" or "No Mask".
* Supports multiple faces in one image.
* Confidence score displayed on detection.
  
**Video File Detection**
* Upload a pre-recorded video file (e.g., MP4, AVI).
* Processes each frame to detect mask status in real-time.
* Outputs a video with annotations for each frame.
  
**Real-Time Detection via Webcam**
* Uses your device's built-in webcam.
* Detects and classifies faces continuously as the camera captures video.
* Displays results with bounding boxes and labels.
  
**Live IP Camera Detection**
* Stream real-time footage from an IP camera using the camera’s RTSP/HTTP URL.
* Applies face and mask detection frame-by-frame.
* Ideal for integrating into a security system or surveillance network.
# Prerequisites
Before running the Face Mask Detection System, make sure your environment is properly set up:
* Python Installation

* Required Python Libraries
Install all dependencies using pip.

* Pre-trained Face Mask Detection Model
You’ll need a trained Keras/TensorFlow model (mask_detector.model) that classifies faces with or without masks.
# Backend
The system uses a deep learning model built with TensorFlow/Keras for classification, and OpenCV for real-time image processing and face detection.
# Frontend
The web interface is built using Streamlit, a Python library that makes it easy to create and deploy web applications with minimal code.
# Contribute
Sakshi welcomes contributions to this project! If you discover a bug, have a feature request, or want to help improve the system, feel free to open an issue or submit a pull request. Please follow the project’s code of conduct and style guide.

# Contact
If you have any questions, suggestions, or collaboration ideas, feel free to reach out to Sakshi at sakshichoudhary129@gmail.com.


