import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import sys
import time

sys.path.append('yolov7')

from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load the YOLOv7 model
@st.cache_resource
def load_model(weights_path):
    model = attempt_load(weights_path, map_location='cpu')  # Load the model to CPU
    print("Model loaded successfully")
    return model

# Function to perform object detection
def detect_objects(image, model, img_size=640):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Ensure correct color space
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
        # Add other class mappings here
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

def main():
    st.title("Poultry Intrusion Detection")
    st.text("Using YOLOv7-tiny with Streamlit")

    # Load the model
    model = load_model('best.pt')  # Update with your trained model path

    # Add buttons for image upload or camera feed
    option = st.selectbox("Choose an option", ["Upload Image", "Real-time Video"])

    if option == "Real-time Video":
        st.text("Using the laptop camera")
        cap = cv2.VideoCapture(0)  # 0 is the default camera
        stframe = st.empty()
        fps_text = st.empty()  # Create an empty container for the FPS display

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video")
                break

            # Perform object detection
            frame = detect_objects(frame, model)

            # Convert OpenCV image to RGB format for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the output in the Streamlit app
            stframe.image(frame, channels="RGB", use_column_width=True)

            # Calculate and display frame rate
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                fps_text.text(f"Frame rate: {fps:.2f} FPS")  # Update the FPS display
                frame_count = 0
                start_time = time.time()

        cap.release()

    elif option == "Upload Image":
        st.text("Upload an image for detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the image file
            image = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB mode
            image = np.array(image)

            # Perform object detection
            image = detect_objects(image, model)

            # Convert OpenCV image to RGB format for display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the numpy array back to PIL image for display
            image = Image.fromarray(image)

            # Display the output in the Streamlit app
            st.image(image, caption='Detected Image', use_column_width=True)

if __name__ == "__main__":
    main()
