# Poultry Intrusion Detection System

## Overview

This project aims to develop a Poultry Intrusion Detection System using YOLOv7-tiny with Streamlit. The system can detect objects in real-time from a laptop camera or uploaded images. It provides an interactive web interface to visualize the detected objects and their confidence scores.
It has 2 Frameworks which you can use first using Streamlit and second using Flask. You can use any one of it setup is same for both just clone the yolov7 repository and define the path correctly

## Features

- **Real-time Video Detection:** Capture and process live video from a laptop camera to detect objects.
- **Image Upload:** Upload images to perform object detection and display results.
- **Frame Rate Display:** Show the frame rate of the video feed for performance monitoring.
- **Object Detection:** Detect and label objects with bounding boxes and confidence scores.

## Requirements

- Python 3.9
- Flask 
- Streamlit
- OpenCV
- PyTorch
- YOLOv7-tiny model and weights

## Setup

Clone the Repository:
```
   git clone <repository-url>
   cd <repository-directory>
```

Clone the yolov7 Repository in the same location as your app.py :
```
!git clone https://github.com/WongKinYiu/yolov7
```

Install Dependencies: Ensure you have the required Python libraries installed:
Run this snippet in the terminal
```
pip install -r requirements.txt
```

Copy code
```
pip install streamlit opencv-python-headless torch torchvision
```
Download YOLOv7-tiny Weights:

Place your YOLOv7-tiny model weights (best.pt) in the project directory.

## Directory Structure:
```
poultry_intrusion_detection/
├── yolov7/
│   ├── models/
│   │   └── experimental.py
│   ├── utils/
│   │   └── general.py
├── app.py
└── best.pt
```

Usage
Running the Streamlit Application
Start the Streamlit App:

```
streamlit run app.py
```

Open Your Browser:

Navigate to http://localhost:8501 to access the application.

Interacting with the Application

## Real-time Video Detection:

Select "Real-time Video" from the dropdown.

The application will access your laptop's camera and display the video feed with detected objects and frame rate information.

## Image Upload:

Select "Upload Image" from the dropdown.

Click the "Choose an image..." button to upload an image file (jpg, jpeg, png).

The application will process the image and display detected objects with bounding boxes and labels.
