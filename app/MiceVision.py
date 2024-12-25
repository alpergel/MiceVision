import streamlit as st
import os
from ultralytics import YOLO, solutions
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
import os
from datetime import datetime
import pandas as pd
from collections import deque
from PIL import ImageDraw, ImageFont
import zipfile
import imageio.v3 as iio
from utils import *

def generateHeatmap(modelPath, video, progress_text, sampleRate):
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    totalNoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    video_name = os.path.splitext(os.path.basename(video))[0]
    path = f"/app/heatmaps/heatmap_output_{video_name}.mp4"
    video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Init heatmap
    heatmap = solutions.Heatmap(
        show=False,
        model=modelPath,
        colormap=cv2.COLORMAP_PARULA,
    )
    for frameCount in range(0, min(int(10 * 60 * fps), totalNoFrames), sampleRate):
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        my_bar2.progress(round((frameCount/totalNoFrames)*100), text=progress_text)
        im0 = heatmap.generate_heatmap(im0)
        video_writer.write(im0)
    video_writer.release()
    return path

def count_groups_of_ones(arr, val):
    count = 0
    in_group = False
    for value in arr:
        if value == val:
            if not in_group:
                count += 1
                in_group = True
        else:
            in_group = False
    return count

def labelNOR(video, sampleRate, yoloInteractor, yoloMouse, yoloLocalizer):
    # Create video capture object
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None  # Return None instead of exit() for better error handling

    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
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

        # Detect time period
        frametime = frameCount * frameT
        framePeriod = np.searchsorted(time_periods, frametime)

        # Check for mouse presence and run YOLO
        mouseCheck = yoloMouse(frame, verbose=False, conf=0.5, max_det = 2, half=True)
        if len(mouseCheck[0].boxes.xyxy.cpu().numpy()) == 2:

            
            # Get Interaction Region Crop
            cropArr = run_interaction(frame, yoloMouse, yoloLocalizer, padding=15)
            
            # If Crops have been generated
            if cropArr is not None:
                for crop in cropArr:
                    # Skip if crop size is 0
                    if len(crop[0]) == 0:
                        continue
                    else:
                        # Convert crop to greyscale
                        gray_crop = cv2.cvtColor(crop[0], cv2.COLOR_BGR2GRAY)
                        #frames.append(gray_crop)
                        
                        # Run YOLO Interaction Detector
                        interaction_results = yoloInteractor(gray_crop, verbose=False)
                        
                        # If Crop is From Left Side
                        if crop[1][0] == 0:
                            # If Interaction Detected
                            if interaction_results[0].probs.top1 == 0:
                                
                                # Add 1 to the count
                                leftArr.append(1)
                                
                                # Increase Interaction Time
                                leftTime[framePeriod] += frameT
                                
                                # Determine Top/Bottom Interaction
                                
                                if crop[1][1] == 0:
                                    objLeftInt.append(0)
                                    leftTimePer[0][0] += frameT
                                else:
                                    objLeftInt.append(1)
                                    leftTimePer[0][1] += frameT
                            else:
                                leftArr.append(0)
                                
                        # If Crop is From Right Side
                        elif crop[1][0] == 1:
                            # If Interaction Detected
                            if interaction_results[0].probs.top1 == 0:
                                
                                # Add 1 to the count
                                rightArr.append(1)
                                
                                # Increase Interaction Time
                                rightTime[framePeriod] += frameT
                                
                                # Determine Top/Bottom Interaction
                                if crop[1][1] == 0:
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
    print(leftArr)
    leftAp = count_groups_of_ones(leftArr, 1)
    rightArr = np.array(rightArr)
    rightAp = count_groups_of_ones(rightArr, 1)
    rightApT = rightAp * frameT
    leftApT = leftAp * frameT
    topObjL = count_groups_of_ones(np.array(objLeftInt),0)
    bottomObjL = count_groups_of_ones(np.array(objLeftInt),1)
    topObjR = count_groups_of_ones(np.array(objRightInt),0)
    bottomObjR = count_groups_of_ones(np.array(objRightInt),1)
    return leftArr, rightArr, leftAp, rightAp, leftApT, rightApT, leftTime, rightTime, topObjL, bottomObjL, topObjR, bottomObjR, frames

start = False

with st.sidebar:
    folder = file_selector()
    if st.button("Submit", type="primary"):
        st.write('You selected `%s`' % folder)
        st.write(f'Running pipeline over {sum([len(files) for r, d, files in os.walk(folder)])} video(s)')
        start = True
    st.divider()
    sampleRate = st.slider("Select a rate to sample images from the videos at. Best Results @ 15 FPS",1,30,15)
    generate_heatmap = st.checkbox("Generate Heatmap", value=False)

    if generate_heatmap:
        st.write("Heatmap will be generated for the interactions.")

# Activate Models
torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
localizerPath = 'Models/yolo11s_OBJ_995_mAP50.pt'
mousePath = 'Models/yolo11s_MOUS_995_mAP50.pt'
interactorPath = 'Models/yolo11l_CLS_95_top1.pt'
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
videos = []

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
                # NOR Operations
                progress_text = f"Running NOR: Video {vidIndex}"
                my_bar = st.progress(0, text=progress_text)
                video = os.path.join(path, file)
                leftArr, rightArr, leftAp, rightAp, leftApT, rightApT, leftTime, rightTime, topObjL, bottomObjL, topObjR, bottomObjR, frames = labelNOR(video, sampleRate, yoloInteractor, yoloMouse, yoloLocalizer)
                print(leftArr)
                data = {
                    "Video Name": [video],
                    "Left Interaction Time Per Period": [leftTime],
                    "Right Interaction Time Per Period": [rightTime],
                    "Left Array": [leftArr],
                    "Right Array": [rightArr],
                    "Left Total Approach": [leftAp],
                    "Right Total Approach": [rightAp],
                    "Left Top Object Interaction": [topObjL],
                    "Left Bottom Object Interaction": [bottomObjL],
                    "Right Top Object Interaction": [topObjR],
                    "Right Bottom Object Interaction": [bottomObjR]
                }
                df = pd.DataFrame(data)
                st.session_state.processed_data['dfTot'] = pd.concat([st.session_state.processed_data['dfTot'], df], ignore_index=True)
                st.session_state.processed_data['globalFrames'].extend(frames)
                my_bar.empty()
                # Heatmap Generation Operations
                if generate_heatmap:
                    progress_text2 = f"Processing Heatmap: Video {vidIndex}"
                    my_bar2 = st.progress(0, text=progress_text2)
                    videos.append(generateHeatmap(mousePath,video,progress_text2, sampleRate))
                    my_bar2.empty()
                vidIndex += 1


    
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
       
        # if st.button("◀ Previous", use_container_width=True):
        #     if (st.session_state.processed_data['image_index'] - 1) % len(pil_images) >= 0:
        #         st.session_state.processed_data['image_index'] = (st.session_state.processed_data['image_index'] - 1) % len(pil_images)
        #     else:
        #         st.session_state.processed_data['image_index'] =  len(pil_images) - 1        
        # if st.button("Next ▶", use_container_width=True):
        #     if (st.session_state.processed_data['image_index'] + 1) % len(pil_images) <= len(pil_images):
        #         st.session_state.processed_data['image_index'] = (st.session_state.processed_data['image_index'] + 1) % len(pil_images)
        #     else:
        #         st.session_state.processed_data['image_index'] = 0
        current_datetime = datetime.now()
        st.download_button(
            label="Download data as CSV",
            data=st.session_state.processed_data['dfTot'].to_csv(),
            file_name=f"{current_datetime}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        # if len(st.session_state.processed_data['incorrect']) > 0:
        #     zip_all_filename = "all_saved_frames.zip"
        #     # Create a button to zip and download all saved frames
        #     if st.button("Zip Saved Frames", use_container_width=True):
        #         with zipfile.ZipFile(zip_all_filename, 'w') as zipf:
        #             for saved_frame in st.session_state.processed_data['incorrect']:
        #                 if os.path.isfile(saved_frame):
        #                     zipf.write(saved_frame, arcname=os.path.basename(saved_frame))
        #         with open(zip_all_filename, 'rb') as f:
        #             st.download_button('Download Zip', f, file_name='Images.zip', use_container_width=True)    
        


