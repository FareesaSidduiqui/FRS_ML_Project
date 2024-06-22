{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a972edd4-b172-4c3f-b4b4-79ec50971d35",
   "metadata": {},
   "source": [
    "## Setup Growth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "95f60b6a-dda7-42c7-b2ff-e524421fa221",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "# Prevent duplicate registrations by checking if GPUs are already configured\n",
    "if not tf.config.list_logical_devices('GPU'):\n",
    "    gpus = tf.config.list_physical_devices('GPU')\n",
    "    if gpus:\n",
    "        try:\n",
    "            for gpu in gpus:\n",
    "                tf.config.set_logical_device_configuration(\n",
    "                    gpu,\n",
    "                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # example: set memory limit to 4GB\n",
    "            logical_gpus = tf.config.list_logical_devices('GPU')\n",
    "            print(len(gpus), \"Physical GPUs,\", len(logical_gpus), \"Logical GPUs\")\n",
    "        except RuntimeError as e:\n",
    "            print(e)\n",
    "    else:\n",
    "        print(\"No GPU found\")\n",
    "else:\n",
    "    print(\"GPUs are already configured\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5588d98a-f7c5-447f-afa9-348ab9f01637",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# DETECTION PART"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c9254257-40d7-4c14-825d-3737436e1cac",
   "metadata": {},
   "source": [
    "#  Imports and Initialization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f0bb51ef-1580-41ec-b140-5d569964f527",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import os\n",
    "from ultralytics import YOLO\n",
    "\n",
    "# Load the YOLOv8n model for face detection\n",
    "model = YOLO('yolov8n-face.pt')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "279a8ded-c366-4319-8e6a-15de1b1f7b3a",
   "metadata": {},
   "source": [
    "# Setting Up Paths and Directories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1064acd9-6ae8-4551-9da6-b73923a990f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Path to the directory where known faces will be stored\n",
    "known_faces_dir = 'known_faces'\n",
    "\n",
    "# Create the directory if it doesn't exist\n",
    "if not os.path.exists(known_faces_dir):\n",
    "    os.makedirs(known_faces_dir)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bbc3b382-ac3d-41c1-946a-fd1a42bb98dc",
   "metadata": {},
   "source": [
    "# Function capture_images(name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c6021d63-203a-483c-943f-6f6654160696",
   "metadata": {},
   "outputs": [],
   "source": [
    "def capture_images(name):\n",
    "    # Create a directory for the new person if it doesn't exist\n",
    "    person_dir = os.path.join(known_faces_dir, name)\n",
    "    if not os.path.exists(person_dir):\n",
    "        os.makedirs(person_dir)\n",
    "    \n",
    "    # Open a connection to the webcam\n",
    "    cap = cv2.VideoCapture(0)  # Change the argument to a file path for video file\n",
    "\n",
    "    if not cap.isOpened():\n",
    "        print(\"Error: Could not open webcam.\")\n",
    "        return\n",
    "\n",
    "    images_captured = 0\n",
    "    while images_captured < 20 and cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        if not ret:\n",
    "            print(\"Error: Failed to capture image.\")\n",
    "            break\n",
    "        \n",
    "        # Convert the frame to RGB (YOLO model expects RGB images)\n",
    "        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        \n",
    "        # Perform inference using YOLOv8n model\n",
    "        results = model(rgb_frame)\n",
    "        \n",
    "        # Convert results to a list of bounding boxes\n",
    "        boxes = results[0].boxes.data.cpu().numpy()  # Get the boxes from the first result\n",
    "        \n",
    "        for box in boxes:\n",
    "            # Extract the coordinates and confidence score\n",
    "            x1, y1, x2, y2, confidence = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]\n",
    "            \n",
    "            # Draw the bounding box on the frame\n",
    "            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)\n",
    "            \n",
    "            # Capture the face ROI\n",
    "            face_roi = frame[y1:y2, x1:x2]\n",
    "            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)\n",
    "            \n",
    "            # Save the captured face image\n",
    "            file_path = os.path.join(person_dir, f'{name}_img{images_captured + 1}.jpg')\n",
    "            cv2.imwrite(file_path, rgb_face)\n",
    "            print(f\"Captured {file_path}\")\n",
    "            images_captured += 1\n",
    "            \n",
    "            # Display the captured face image\n",
    "            cv2.imshow('Captured Face', rgb_face)\n",
    "            cv2.waitKey(100)  # Pause to display the image\n",
    "\n",
    "        # Display the frame with bounding boxes\n",
    "        cv2.imshow('Face Capture', frame)\n",
    "        \n",
    "        # Exit the loop on 'q' key press\n",
    "        if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "    # Release the webcam and close all OpenCV windows\n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f956ec4a-acfd-472f-ad49-567d4e864659",
   "metadata": {},
   "source": [
    "# Input and Execution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8f6de664-4020-4ce2-9a37-486e3a16a022",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "0: 480x640 1 face, 202.1ms\n",
      "Speed: 0.0ms preprocess, 202.1ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img1.jpg\n",
      "\n",
      "0: 480x640 1 face, 129.7ms\n",
      "Speed: 0.0ms preprocess, 129.7ms inference, 10.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img2.jpg\n",
      "\n",
      "0: 480x640 1 face, 116.7ms\n",
      "Speed: 4.6ms preprocess, 116.7ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img3.jpg\n",
      "\n",
      "0: 480x640 1 face, 160.1ms\n",
      "Speed: 0.0ms preprocess, 160.1ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img4.jpg\n",
      "\n",
      "0: 480x640 1 face, 168.0ms\n",
      "Speed: 3.0ms preprocess, 168.0ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img5.jpg\n",
      "\n",
      "0: 480x640 1 face, 163.3ms\n",
      "Speed: 3.5ms preprocess, 163.3ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img6.jpg\n",
      "\n",
      "0: 480x640 1 face, 162.7ms\n",
      "Speed: 2.2ms preprocess, 162.7ms inference, 1.5ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img7.jpg\n",
      "\n",
      "0: 480x640 1 face, 166.9ms\n",
      "Speed: 0.0ms preprocess, 166.9ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img8.jpg\n",
      "\n",
      "0: 480x640 1 face, 141.6ms\n",
      "Speed: 0.9ms preprocess, 141.6ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img9.jpg\n",
      "\n",
      "0: 480x640 1 face, 127.7ms\n",
      "Speed: 0.0ms preprocess, 127.7ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img10.jpg\n",
      "\n",
      "0: 480x640 1 face, 145.7ms\n",
      "Speed: 0.0ms preprocess, 145.7ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img11.jpg\n",
      "\n",
      "0: 480x640 1 face, 149.1ms\n",
      "Speed: 0.0ms preprocess, 149.1ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img12.jpg\n",
      "\n",
      "0: 480x640 1 face, 129.1ms\n",
      "Speed: 0.0ms preprocess, 129.1ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img13.jpg\n",
      "\n",
      "0: 480x640 1 face, 127.1ms\n",
      "Speed: 0.0ms preprocess, 127.1ms inference, 3.7ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img14.jpg\n",
      "\n",
      "0: 480x640 1 face, 121.0ms\n",
      "Speed: 5.4ms preprocess, 121.0ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img15.jpg\n",
      "\n",
      "0: 480x640 1 face, 154.6ms\n",
      "Speed: 0.0ms preprocess, 154.6ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img16.jpg\n",
      "\n",
      "0: 480x640 1 face, 139.9ms\n",
      "Speed: 0.0ms preprocess, 139.9ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img17.jpg\n",
      "\n",
      "0: 480x640 1 face, 129.1ms\n",
      "Speed: 0.0ms preprocess, 129.1ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img18.jpg\n",
      "\n",
      "0: 480x640 1 face, 128.0ms\n",
      "Speed: 0.0ms preprocess, 128.0ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img19.jpg\n",
      "\n",
      "0: 480x640 1 face, 141.3ms\n",
      "Speed: 3.5ms preprocess, 141.3ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "Captured known_faces\\cs211048\\cs211048_img20.jpg\n"
     ]
    }
   ],
   "source": [
    "# Prompt user to enter the name of the new person\n",
    "name = input(\"Enter the name of the new person: \")\n",
    "\n",
    "# Call the function to capture images\n",
    "capture_images(name)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b87b8e5d-1554-4cbd-aa97-df569824514c",
   "metadata": {},
   "source": [
    "# RECOGNITION PART"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "171cd17b-cbcc-4550-b341-ef8a74c398f9",
   "metadata": {},
   "source": [
    "#  Imports and Initialization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "63b6ed19-580f-44a3-86b3-2724d3b5e0fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import face_recognition\n",
    "import numpy as np\n",
    "import os\n",
    "from ultralytics import YOLO\n",
    "import pandas as pd\n",
    "from datetime import datetime\n",
    "import pyttsx3\n",
    "\n",
    "# Initialize global variables\n",
    "known_faces_dir = 'known_faces'\n",
    "attendance_file = 'attendance.csv'\n",
    "engine = pyttsx3.init()\n",
    "voice_id = \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0\"\n",
    "engine.setProperty('voice', voice_id)\n",
    "known_face_encodings = []\n",
    "known_face_names = []\n",
    "unknown_detected = False  # Initialize unknown_detected\n",
    "recorded_names = set()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b05ccfc5-6d91-4bec-a411-a62001d7e61d",
   "metadata": {},
   "source": [
    "# Functions Definitions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e3ceb850-00b5-462c-b760-41565be7d5dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_known_faces():\n",
    "    global known_face_encodings, known_face_names\n",
    "    for root, _, files in os.walk(known_faces_dir):\n",
    "        for filename in files:\n",
    "            if filename.endswith(('.png', '.jpg', '.jpeg')):\n",
    "                filepath = os.path.join(root, filename)\n",
    "                img = face_recognition.load_image_file(filepath)\n",
    "                encodings = face_recognition.face_encodings(img)\n",
    "                if encodings:\n",
    "                    encoding = encodings[0]\n",
    "                    known_face_encodings.append(encoding)\n",
    "                    name = os.path.splitext(filename)[0].split('_')[0]\n",
    "                    known_face_names.append(name)\n",
    "\n",
    "def mark_attendance(name):\n",
    "    if not os.path.isfile(attendance_file):\n",
    "        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])\n",
    "        df.to_csv(attendance_file, index=False)\n",
    "\n",
    "    if name != \"Unknown\":\n",
    "        now = datetime.now()\n",
    "        date_str = now.strftime('%Y-%m-%d')\n",
    "        time_str = now.strftime('%H:%M:%S')\n",
    "        df = pd.DataFrame([[name, date_str, time_str]], columns=['Name', 'Date', 'Time'])\n",
    "        df.to_csv(attendance_file, mode='a', header=False, index=False)\n",
    "\n",
    "        engine.say(f\"{name}, attendance is marked\")\n",
    "        engine.runAndWait()\n",
    "\n",
    "def initialize_yolo():\n",
    "    return YOLO('yolov8n-face.pt')\n",
    "\n",
    "def recognize_faces(frame, model):\n",
    "    global unknown_detected\n",
    "    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "    results = model(rgb_frame)\n",
    "    boxes = results[0].boxes.data.cpu().numpy()\n",
    "    face_locations = []\n",
    "    face_names = []\n",
    "    \n",
    "    for box in boxes:\n",
    "        x1, y1, x2, y2, confidence = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]\n",
    "        face_locations.append((y1, x2, y2, x1))\n",
    "        face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[-1]])\n",
    "        name = \"Unknown\"\n",
    "        \n",
    "        if face_encodings:\n",
    "            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])\n",
    "            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])\n",
    "            \n",
    "            if len(face_distances) > 0:\n",
    "                best_match_index = np.argmin(face_distances)\n",
    "                if matches[best_match_index]:\n",
    "                    name = known_face_names[best_match_index]\n",
    "                    if name not in recorded_names:\n",
    "                        mark_attendance(name)\n",
    "                        recorded_names.add(name)\n",
    "                else:\n",
    "                    if not unknown_detected:\n",
    "                        print(\"Unknown person detected. Attendance cannot be marked.\")\n",
    "                        engine.say(\"Unknown person detected. Attendance cannot be marked.\")\n",
    "                        engine.runAndWait()\n",
    "                        unknown_detected = True\n",
    "        \n",
    "        face_names.append(name)\n",
    "        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)\n",
    "        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)\n",
    "    \n",
    "    return frame\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4c6c4d9d-588d-426f-9fcd-a9a17d1c5b54",
   "metadata": {},
   "source": [
    "# Function Recognition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "41c97193-ce4c-4c00-abc8-607e187dfbab",
   "metadata": {},
   "outputs": [],
   "source": [
    "def start_recognition():\n",
    "    load_known_faces()\n",
    "    model = initialize_yolo()\n",
    "    \n",
    "    cap = cv2.VideoCapture(0)\n",
    "    if not cap.isOpened():\n",
    "        print(\"Error: Could not open webcam.\")\n",
    "        exit()\n",
    "\n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        if not ret:\n",
    "            print(\"Error: Failed to capture image.\")\n",
    "            break\n",
    "        \n",
    "        frame = recognize_faces(frame, model)\n",
    "        cv2.imshow('Face Recognition', frame)\n",
    "        \n",
    "        if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "    cap.release()\n",
    "    cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ea402a6f-6ee9-40dc-a532-5b934d1da538",
   "metadata": {},
   "source": [
    "# Main Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "22637532-a8ae-4718-be28-caa3acb47a89",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "0: 480x640 1 face, 168.2ms\n",
      "Speed: 0.0ms preprocess, 168.2ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 156.0ms\n",
      "Speed: 2.7ms preprocess, 156.0ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 141.2ms\n",
      "Speed: 0.0ms preprocess, 141.2ms inference, 5.2ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 127.5ms\n",
      "Speed: 0.0ms preprocess, 127.5ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 127.5ms\n",
      "Speed: 0.0ms preprocess, 127.5ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 118.5ms\n",
      "Speed: 3.3ms preprocess, 118.5ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 129.4ms\n",
      "Speed: 2.0ms preprocess, 129.4ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 127.4ms\n",
      "Speed: 0.0ms preprocess, 127.4ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 124.5ms\n",
      "Speed: 0.0ms preprocess, 124.5ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 116.4ms\n",
      "Speed: 0.0ms preprocess, 116.4ms inference, 3.5ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 127.0ms\n",
      "Speed: 0.0ms preprocess, 127.0ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n",
      "\n",
      "0: 480x640 1 face, 165.9ms\n",
      "Speed: 2.6ms preprocess, 165.9ms inference, 0.0ms postprocess per image at shape (1, 3, 480, 640)\n"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    input_name = input(\"Enter your ID: \")\n",
    "    start_recognition()\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f31274ed-172d-49f2-bd9a-cb59d29a7fba",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
