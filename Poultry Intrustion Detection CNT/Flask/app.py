import time
from flask import Flask, render_template, Response, request
import torch
import cv2
import numpy as np
from PIL import Image
import base64
import sys
import os

# Add the yolov7 directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Streamlit', 'yolov7')))

from models.experimental import attempt_load  # Import from yolov7
from utils.general import non_max_suppression  # Import from yolov7

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv7 model
def load_model(weights_path='best.pt'):
    try:
        model = attempt_load(weights_path, map_location='cpu')  # Load model
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully.")  # Debug print statement
        return model
    except Exception as e:
        print(f"Error loading model: {e}")  # Print any errors
        sys.exit(1)  # Exit if model cannot be loaded

model = load_model()

# Function to perform object detection
print_detection = True

def detect_objects(image, model, img_size=640):
    global print_detection  # Access the global flag
    if print_detection:
        print("Performing object detection...")
        print_detection = False

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (img_size, img_size))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img /= 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)

    if isinstance(pred, tuple):
        pred = pred[0]

    detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # Define class names mapping
    class_names = {
        0: 'Cat',
        1: 'Chicken',
        2: 'Dog',
        3: 'Human',
        4: 'Snake',
    }

    if detections is not None and len(detections):
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(cls)
            class_name = class_names.get(cls, 'Unknown')
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name} Conf: {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        option = request.form.get('option')
        print(f"Selected Option: {option}")  # Debug print statement

        if option == 'Upload Image':
            if 'file' not in request.files:
                print("No file part in request.")  # Debug print statement
                return render_template('index.html', option=option, error='No file part')
            file = request.files['file']
            if file.filename == '':
                print("No file selected.")  # Debug print statement
                return render_template('index.html', option=option, error='No selected file')

            try:
                image = Image.open(file.stream).convert('RGB')
                image_np = np.array(image)
                result_image = detect_objects(image_np, model)

                # Convert result image to base64 string for rendering
                _, img_encoded = cv2.imencode('.jpg', result_image)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')

                return render_template('index.html', option=option, image_data=img_base64)
            except Exception as e:
                print(f"Error processing file: {e}")  # Debug print statement
                return render_template('index.html', option=option, error=f'Error processing file: {str(e)}')

    return render_template('index.html', option='Real-time Video')

# Route for video streaming
def gen_frames(n=20):  # n is the frame interval
    cap = cv2.VideoCapture(0)  # 0 for the default camera
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    frame_counter = 0
    prev_time = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture video frame.")
            break

        # Calculate frame rate
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Print frame rate every 10 seconds
        if int(curr_time) % 10 == 0:
            print(f"Frame Rate: {fps:.2f} FPS")

        # Process every nth frame
        if frame_counter % n == 0:
            frame = detect_objects(frame, model)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_counter += 1

    cap.release()
    print("Video feed ended.")  # Debug print statement


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
