import streamlit as st
import os
from ultralytics import YOLO, solutions
import numpy as np
import cv2
import PIL
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
import os
from datetime import datetime
import pandas as pd
import random
from collections import deque
import string
from PIL import ImageDraw, ImageFont
import zipfile
import imageio.v3 as iio


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a folder', filenames)
    return os.path.join(folder_path, selected_filename)

def centroid(detection):
    if len(detection.boxes) > 0:
        box = detection.boxes[0].xyxy[0].cpu().numpy()  # Ensure we're working with numpy arrays
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return None

def generate_random_name(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Count Number of Approaches
def count_groups_of_ones(lst, cons):
    counter = 0  # Initialize the counter
    i = 0  # Start iterating from the first index
    while i < len(lst):
        # Check if the current element is 0 and at least the next three elements contain a 1
        if lst[i] == 0:
            i += 1  # Move past the 0
            start = i  # Mark the start of potential consecutive 1's
            # Count the number of consecutive 1's following the 0
            while i < len(lst) and lst[i] == 1:
                i += 1
            # If we found at least three consecutive 1's after a 0, increment the counter
            if i - start >= cons:
                counter += 1
        else:
            i += 1  # Move to the next element if the current one is not 0
    return counter

# Count per object Approach
def count_obj_approach(lst):
  # Bottom is index 0, top index 1
  top = lst.count(1)
  bottom = len(lst) - top
  return [bottom, top]

def get_centroid(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

def get_bounding_box(detection):
    if len(detection.boxes) > 0:
        return detection.boxes[0].xyxy[0].cpu().numpy()
    return None

def create_convex_hull_box(box1, box2):
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    return np.array([x_min, y_min, x_max, y_max])

def process_crop(image, large_box, padding):
    large_box[0] = max(0, large_box[0] - padding)
    large_box[1] = max(0, large_box[1] - padding)
    large_box[2] = min(image.shape[1], large_box[2] + padding)
    large_box[3] = min(image.shape[0], large_box[3] + padding)

    # Ensure the box is within image dimensions
    height, width = image.shape[:2]
    large_box = np.array([
        max(0, large_box[0]),
        max(0, large_box[1]),
        min(width, large_box[2]),
        min(height, large_box[3])
    ]).astype(int)

    # Crop the image
    cropped_image = image[large_box[1]:large_box[3], large_box[0]:large_box[2]]
    
    
    if cropped_image.size == 0 or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        return []
    
    cropped_image = cv2.resize(cropped_image, (256, 256))
    return [cropped_image]
    
def run_interaction(left_frame, right_frame, mouse_model, object_model, padding=10):
    cropArr = []
    left_frame = cv2.resize(left_frame, (416,416))
    right_frame = cv2.resize(right_frame, (416,416))
    
    # Detect Mice
    mouseL = mouse_model(left_frame, verbose=False,  max_det = 1)[0]
    mouseR = mouse_model(right_frame, verbose=False,  max_det = 1)[0]
    
    if len(mouseL) > 0 and len(mouseR) > 0:
        # Detect objects
        objectsL = object_model(left_frame, verbose=False, max_det = 2)
        lBoxes = []
        for obj in objectsL:
            lBoxes.append(obj.boxes.xyxy.cpu().numpy())
        objectsR = object_model(right_frame, verbose=False, max_det = 2)
        rBoxes = []
        for obj in objectsR:
            rBoxes.append(obj.boxes.xyxy.cpu().numpy())
        
        # Sort Sided Object Arrays by Y coordinate
        lBoxes = sorted(lBoxes[0], key=lambda o: get_centroid(o)[1])
        rBoxes = sorted(rBoxes[0], key=lambda o: get_centroid(o)[1])
        
        # Process left side
        mouse_centroid = get_centroid(mouseL[0].boxes.xyxy.cpu().numpy()[0])
        object_centroids = []
        for box in lBoxes:
            object_centroids.append(get_centroid(box))
        distances = [np.linalg.norm(mouse_centroid - obj_centroid) for obj_centroid in object_centroids]
        if len(distances) > 1:
            chosen_object_box = lBoxes[np.argmin(distances)]
            large_box = create_convex_hull_box(mouseL[0].boxes.xyxy.cpu().numpy()[0], chosen_object_box)
            cropArr.append([process_crop(left_frame, large_box, padding),0,np.argmin(distances)])

        # Process right side
        mouse_centroid = get_centroid(mouseR[0].boxes.xyxy.cpu().numpy()[0])
        object_centroids = []
        for box in rBoxes:
            object_centroids.append(get_centroid(box))
        distances = [np.linalg.norm(mouse_centroid - obj_centroid) for obj_centroid in object_centroids]
        if len(distances) > 1:
            chosen_object_box = rBoxes[np.argmin(distances)]
            large_box = create_convex_hull_box(mouseR[0].boxes.xyxy.cpu().numpy()[0], chosen_object_box)
            cropArr.append([process_crop(right_frame, large_box, padding),1,np.argmin(distances)])

        return cropArr if cropArr else None
    else:
        return None


def labelNOR(video, sampleRate, yoloInteractor, yoloMouse, yoloLocalizer):
    # Create video capture object
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None  # Return None instead of exit() for better error handling

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    totalNoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameT = totalNoFrames / (fps * totalNoFrames)  # Time per frame

    # Initialize counters and arrays
    leftTime = np.zeros(3, dtype=np.float32)
    rightTime = np.zeros(3, dtype=np.float32)
    leftTimePer = [[0.0,0.0],[0.0,0.0],[0.0,0.0]]
    rightTimePer = [[0.0,0.0],[0.0,0.0],[0.0,0.0]]
    leftArr = []
    rightArr = []
    objLeftInt = []
    objRightInt = []
    frames = []
    splitframes = []

    # Pre-compute time periods
    T1, T2, T3 = 150, 300, 600
    time_periods = np.array([T1, T2, T3])

    # Use deques for efficient append and pop operations
    leftArr = deque()
    rightArr = deque()

    for frameCount in range(0, min(int(10 * 60 * fps), totalNoFrames), sampleRate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameCount)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (416,416))

        # Update progress
        my_bar.progress(round((frameCount/totalNoFrames)*100), text=progress_text)

        # Split frame
        midpoint = frame.shape[1] // 2

        # Detect time period
        frametime = frameCount * frameT
        framePeriod = np.searchsorted(time_periods, frametime)

        # Check for mouse presence and run YOLO
        mouseCheck = yoloMouse(frame, verbose=False, conf=0.5, max_det = 2, half=True)
        if len(mouseCheck[0].boxes.xyxy.cpu().numpy()) == 2:
            
            # Split Image
            left_frame = frame[:, :midpoint]
            right_frame = frame[:, midpoint:]
            
            # Get Interaction Region Crop
            cropArr = run_interaction(left_frame, right_frame, yoloMouse, yoloLocalizer, padding=15)
            if cropArr is not None:
                for crop in cropArr:
                    # Skip if crop size is 0
                    if len(crop[0]) == 0:
                        continue
                    else:
                        # Convert crop to greyscale
                        gray_crop = cv2.cvtColor(crop[0][0], cv2.COLOR_BGR2GRAY)
                        frames.append(gray_crop)
                        
                        # Run YOLO Interaction Detector
                        interaction_results = yoloInteractor(gray_crop, verbose=False)
                        
                        # If Crop is From Left Side
                        if crop[1] == 0:
                            # If Interaction Detected
                            if interaction_results[0].probs.top1 == 0:
                                
                                # Add 1 to the count
                                leftArr.append(1)
                                
                                # Increase Interaction Time
                                leftTime[framePeriod] += frameT
                                
                                # Determine Top/Bottom Interaction
                                if crop[2] == 0:
                                    objLeftInt.append(0)
                                    leftTimePer[0][0] += frameT
                                else:
                                    objLeftInt.append(1)
                                    leftTimePer[0][1] += frameT
                            else:
                                leftArr.append(0)
                                
                        # If Crop is From Right Side
                        elif crop[1] == 1:
                            # If Interaction Detected
                            if interaction_results[0].probs.top1 == 0:
                                
                                # Add 1 to the count
                                rightArr.append(1)
                                
                                # Increase Interaction Time
                                rightTime[framePeriod] += frameT
                                
                                # Determine Top/Bottom Interaction
                                if crop[2] == 0:
                                    objRightInt.append(0)
                                    rightTimePer[0][0] += frameT
                                else:
                                    objRightInt.append(1)
                                    rightTimePer[0][1] += frameT

                            else:
                                rightArr.append(0)

    cap.release()

    # Compute final statistics
    leftArr = np.array(leftArr)
    leftAp = np.sum(leftArr)
    rightArr = np.array(rightArr)
    rightAp = np.sum(rightArr)
    rightApT = rightAp * frameT
    leftApT = leftAp * frameT
    topObjL = objLeftInt.count(0)
    bottomObjL = objLeftInt.count(1)
    topObjR = objRightInt.count(0)
    bottomObjR = objRightInt.count(1)
    return leftArr, rightArr, leftAp, rightAp, leftApT, rightApT, leftTime, rightTime, topObjL, bottomObjL, topObjR, bottomObjR, frames

start = False

with st.sidebar:
    folder = file_selector()
    if st.button("Submit", type="primary"):
        st.write('You selected `%s`' % folder)
        st.write(f'Running pipeline over {sum([len(files) for r, d, files in os.walk(folder)])} video(s)')
        start = True
    st.divider()
    sampleRate = st.slider("Select a rate to sample images from the videos at",1,1000,3)

# Activate Models
torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
localizerPath = 'Models/yolov9_objectLocalizer.pt'
mousePath = 'Models/mouse_detection_yolov9c.pt'
interactorPath = 'Models/YOLOV8_INTERACT.pt'
yoloLocalizer = YOLO(localizerPath).to(device)
yoloMouse = YOLO(mousePath).to(device)
yoloInteractor = YOLO(interactorPath).to(device)

# Setup Batch Inference
video_extensions = ['.mp4', '.avi', '.MOV', '.mkv']
column = ["VideoName", "leftTime", "rightTime", "leftAp", "rightAp", "leftApT", "rightApT", "leftObj", "rightObj"]
dfTot = pd.DataFrame(columns=column)
vidIndex = 1

# Initialize Frame List
globalFrames = []
all_left_arr = []
all_right_arr = []

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {
        'dfTot': pd.DataFrame(),
        'globalFrames': [],
        'all_left_arr': [],
        'all_right_arr': [],
        'image_index': 0,
        'processing_complete': False,
        'incorrect': [],
    }
    
if start and not st.session_state.processed_data['processing_complete']:    
    for monthFolder in os.listdir(folder):
        path = os.path.join(folder, monthFolder)

        for file in os.listdir(path):
            if any(file.endswith(ext) for ext in video_extensions):
                progress_text = f"Processing Video: {vidIndex}"
                my_bar = st.progress(0, text=progress_text)
                video = os.path.join(path, file)
                leftArr, rightArr, leftAp, rightAp, leftApT, rightApT, leftTime, rightTime, topObjL, bottomObjL, topObjR, bottomObjR, frames = labelNOR(video, sampleRate, yoloInteractor, yoloMouse, yoloLocalizer)
                data = {
                    "Video Name": [video],
                    "Left Interaction Time Per Period": [leftTime],
                    "Right Interaction Time Per Period": [rightTime],
                    "Left Total Approach": [leftAp],
                    "Right Total Approach": [rightAp],
                    "Left Top Object Interaction": [topObjL],
                    "Left Bottom Object Interaction": [bottomObjL],
                    "Right Top Object Interaction": [topObjR],
                    "Right Bottom Object Interaction": [bottomObjR]
                }
                df = pd.DataFrame(data)
                st.session_state.processed_data['dfTot'] = pd.concat([st.session_state.processed_data['dfTot'], df], ignore_index=True)
                vidIndex += 1
                st.session_state.processed_data['globalFrames'].extend(frames)
                my_bar.empty()
    
    st.session_state.processed_data['processing_complete'] = True
    
    
if st.session_state.processed_data['processing_complete']:
    # Convert frames to PIL Images
    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in st.session_state.processed_data['globalFrames']]

    # Use session state to keep track of the current image index
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    # Create three columns for layout
    left,center_col,right = st.columns([1,5,1])

    # Display the current image in the center column
    with center_col:
        
        if pil_images:
            current_image = pil_images[st.session_state.processed_data['image_index']]
            st.image(current_image, use_column_width=True)
        else:
            st.write("No frames to display.")
        
       
        if st.button("◀ Previous", use_container_width=True):
            if (st.session_state.processed_data['image_index'] - 1) % len(pil_images) >= 0:
                st.session_state.processed_data['image_index'] = (st.session_state.processed_data['image_index'] - 1) % len(pil_images)
            else:
                st.session_state.processed_data['image_index'] =  len(pil_images) - 1        
        if st.button("Next ▶", use_container_width=True):
            if (st.session_state.processed_data['image_index'] + 1) % len(pil_images) <= len(pil_images):
                st.session_state.processed_data['image_index'] = (st.session_state.processed_data['image_index'] + 1) % len(pil_images)
            else:
                st.session_state.processed_data['image_index'] = 0
        current_datetime = datetime.now()
        st.download_button(
            label="Download data as CSV",
            data=st.session_state.processed_data['dfTot'].to_csv(),
            file_name=f"{current_datetime}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if len(st.session_state.processed_data['incorrect']) > 0:
            zip_all_filename = "all_saved_frames.zip"
            # Create a button to zip and download all saved frames
            if st.button("Zip Saved Frames", use_container_width=True):
                with zipfile.ZipFile(zip_all_filename, 'w') as zipf:
                    for saved_frame in st.session_state.processed_data['incorrect']:
                        if os.path.isfile(saved_frame):
                            zipf.write(saved_frame, arcname=os.path.basename(saved_frame))
                with open(zip_all_filename, 'rb') as f:
                    st.download_button('Download Zip', f, file_name='Images.zip', use_container_width=True)    
        


