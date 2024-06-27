# %% [markdown]
# ## Setup Growth

# %%
import tensorflow as tf

# Prevent duplicate registrations by checking if GPUs are already configured
if not tf.config.list_logical_devices('GPU'):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # example: set memory limit to 4GB
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found")
else:
    print("GPUs are already configured")


# %% [markdown]
# # DETECTION PART

# %% [markdown]
# #  Imports and Initialization

# %%
import cv2
import os
from ultralytics import YOLO

# Load the YOLOv8n model for face detection
model = YOLO('yolov8n-face.pt')


# %% [markdown]
# # Setting Up Paths and Directories

# %%
# Path to the directory where known faces will be stored
known_faces_dir = 'known_faces'

# Create the directory if it doesn't exist
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)


# %% [markdown]
# # Function capture_images(name)

# %%
def capture_images(name):
    # Create a directory for the new person if it doesn't exist
    person_dir = os.path.join(known_faces_dir, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)  # Change the argument to a file path for video file

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    images_captured = 0
    while images_captured < 20 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Convert the frame to RGB (YOLO model expects RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference using YOLOv8n model
        results = model(rgb_frame)
        
        # Convert results to a list of bounding boxes
        boxes = results[0].boxes.data.cpu().numpy()  # Get the boxes from the first result
        
        for box in boxes:
            # Extract the coordinates and confidence score
            x1, y1, x2, y2, confidence = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Capture the face ROI
            face_roi = frame[y1:y2, x1:x2]
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Save the captured face image
            file_path = os.path.join(person_dir, f'{name}_img{images_captured + 1}.jpg')
            cv2.imwrite(file_path, rgb_face)
            print(f"Captured {file_path}")
            images_captured += 1
            
            # Display the captured face image
            cv2.imshow('Captured Face', rgb_face)
            cv2.waitKey(100)  # Pause to display the image

        # Display the frame with bounding boxes
        cv2.imshow('Face Capture', frame)
        
        # Exit the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# %% [markdown]
# # Input and Execution

# %%
# Prompt user to enter the name of the new person
# while True:
name = input("Enter the name of the new ID: ")
#     if not os.path.exists(os.path.join(known_faces_dir, name)):
#         break
#     print(f"Error: Directory for {name} already exists. Please choose a different name.")

# Call the function to capture images
capture_images(name)


# %% [markdown]
# # RECOGNITION PART

# %% [markdown]
# #  Imports and Initialization

# %%
import cv2
import face_recognition
import numpy as np
import os
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import pyttsx3

# Initialize global variables
known_faces_dir = 'known_faces'
attendance_file = 'attendance.csv'
engine = pyttsx3.init()
voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
engine.setProperty('voice', voice_id)
known_face_encodings = []
known_face_names = []
unknown_detected = False  # Initialize unknown_detected
recorded_names = set()

# %% [markdown]
# # Functions Definitions

# %%
def load_known_faces():
    global known_face_encodings, known_face_names
    for root, _, files in os.walk(known_faces_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, filename)
                img = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    encoding = encodings[0]
                    known_face_encodings.append(encoding)
                    name = os.path.splitext(filename)[0].split('_')[0]
                    known_face_names.append(name)

def mark_attendance(name):
    if not os.path.isfile(attendance_file):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        df.to_csv(attendance_file, index=False)

    if name != "Unknown":
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        df = pd.DataFrame([[name, date_str, time_str]], columns=['Name', 'Date', 'Time'])
        df.to_csv(attendance_file, mode='a', header=False, index=False)

        engine.say(f"{name}, attendance is marked")
        engine.runAndWait()

def initialize_yolo():
    return YOLO('yolov8n-face.pt')

def recognize_faces(frame, model):
    global unknown_detected
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    boxes = results[0].boxes.data.cpu().numpy()
    face_locations = []
    face_names = []
    
    for box in boxes:
        x1, y1, x2, y2, confidence = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        face_locations.append((y1, x2, y2, x1))
        face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[-1]])
        name = "Unknown"
        
        if face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name not in recorded_names:
                        mark_attendance(name)
                        recorded_names.add(name)
                else:
                    if not unknown_detected:
                        print("Unknown person detected. Attendance cannot be marked.")
                        engine.say("Unknown person detected. Attendance cannot be marked.")
                        engine.runAndWait()
                        unknown_detected = True
        
        face_names.append(name)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    return frame


# %% [markdown]
# # Function Recognition

# %%
def start_recognition():
    load_known_faces()
    model = initialize_yolo()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        frame = recognize_faces(frame, model)
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# %% [markdown]
# # Main Function

# %%
def main():
    input_name = input("Enter your ID: ")
    start_recognition()

if __name__ == "__main__":
    main()

# %%



