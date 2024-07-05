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
import math
import os
from datetime import datetime
import pandas as pd
from roboflow import Roboflow



def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a folder', filenames)
    return os.path.join(folder_path, selected_filename)

def centroid(results):
    # Assuming results.boxes.xyxy.cpu().numpy() returns a 2D array of shape (n_boxes, 4)
    coords = results.boxes.xyxy.cpu().numpy()

    # Check if there are any boxes detected.
    if coords.shape[0] == 0:
        return None  # No boxes detected.

    x1, y1, x2, y2 = coords[0]  # Access the first box coordinates.
    centroid_obj = (int((x1 + x2) // 2), int((y1 + y2) // 2))
    return centroid_obj

# Approach Detection Functions
def sortDist(l):
  return l[0]

def sortData(d):
  return d[1]

def sortByY(l):
  return l[1]


# Pipeline Function
def calcDist(centroidM, centroidO):
    distList = []
    # Check Dist From Points
    for l in range(len(centroidO)):
      distList.append([math.dist(centroidM, centroidO[l][0]),centroidO[l][1]])
      # Sort List
      distList.sort(key=sortDist)
    # Return Distance
    return distList

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

def labelNOR(vidIndex, video, sampleRate, yoloInteractor, yoloMouse, yoloLocalizer, myBar):
  # Initialize Roboflow
  rf = Roboflow(api_key="FQmBgtlhcDpO84Vn7Cuv")
  project = rf.workspace("conboylablabeling").project("mouseinteraction")
  version = project.version(4)
  
  # create video capture object
  cap = cv2.VideoCapture(video)
  if not cap.isOpened():
    exit()

  # count the number of frames
  w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
  totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  durationInSeconds = totalNoFrames // fps

  # Calculate Time Per Frame
  frameT = durationInSeconds/totalNoFrames

  # Establish Total Interact Counters (In Seconds)
  leftTime = [0.0,0.0,0.0]
  rightTime = [0.0,0.0,0.0]

  # Seperate Into T1, T2, T3 (first 2:30 min, next 2:30 min, and last 5 min of each 10 min video)
  T1, T2, T3 = 150, 300, 600
  frameCount = 0

  # Declare ARR
  leftArr = []
  rightArr = []
  objArrL, objArrR = [] , []
  objLeftInt, objRightInt = [] , []
  frames = []
  
  # MouseCheck
  firstObj = False
  duration = 10 * 60  # 10 minutes in seconds
  start_time = None
  
  # Init heatmap
  #video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
  #heatmap_obj = solutions.Heatmap(
  #    colormap=cv2.COLORMAP_PARULA,
  #    view_img=True,
  #    shape="circle",
  #    classes_names=yoloMouse.names,
  #)

  while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        break
    
    # Calculate the midpoint
    midpoint = frame.shape[1] // 2

    # Split the frame into left and right halves
    left_frame = frame[:, :midpoint]
    right_frame = frame[:, midpoint:]
    
    # Progress Bar
    my_bar.progress(round((frameCount/(totalNoFrames))*100), text=progress_text)
    # Only Run Pipeline over every sample rate one
    if frameCount % sampleRate == 0:
      
      # Append Frames
      frames.append(frame)
      
      # Progress Update
      if frameCount % 1000 == 0:
        print(f"Processing: {frameCount}/{totalNoFrames} Completed {round((frameCount/(totalNoFrames))*100)}%")

      # Detect which Time Period its in
      frametime = frameCount *frameT
      framePeriod = "T1" if frametime < T2 else ("T2" if frametime < T3 else "T3")

      # Check if mouse present
      mouseCheck = yoloMouse([left_frame,right_frame], verbose = False, conf = 0.65, half=True)
      #tracks = yoloMouse.track(left_frame, persist=True, show=False)
      #hm = heatmap_obj.generate_heatmap(left_frame, tracks)
      #video_writer.write(hm)
      
      # Run YOLO Interaction
      if len(mouseCheck[0].boxes) > 0 or len(mouseCheck[1].boxes) > 0:

        # Calculate Mouse Centroid
        lC = centroid(mouseCheck[0])
        rC = centroid(mouseCheck[1])

        # Run Object Localizer
        if firstObj == False:
          objects = yoloLocalizer([left_frame,right_frame], verbose = False, conf = 0.4, half=True)
          for i in range(len(objects[0].boxes)):
            objArrL.append([centroid(objects[0][i])])
          for j in range(len(objects[1].boxes)):
            objArrR.append([centroid(objects[1][j])])
          if objArrR == [] or objArrL == []:
            continue
          else:
            # Sort Objects
            sorted(objArrL , key=lambda k: [k[0][1], k[0][0]])
            for l in range(len(objArrL)):
              objArrL[l].append(l)
            sorted(objArrR , key=lambda k: [k[0][1], k[0][0]])
            for k in range(len(objArrR)):
              objArrR[k].append(k)
            firstObj = True

        # Run Interaction Detector
        results = yoloInteractor([left_frame,right_frame], verbose=False)

        # Left
        if results[0].probs.top1 == 0:
          # Calculate Object Being Interacted with
          if len(objArrL) > 1 and lC is not None:
            objL = calcDist(lC,objArrL)
            if objL[0][0] < objL[1][0]:
              objLeftInt.append(objL[0][1])
            else:
              objLeftInt.append(objL[0][1])
          # Calculate Period
          leftArr.append(1)
          if framePeriod == "T1":
            leftTime[0] += frameT
          elif framePeriod == "T2":
            leftTime[1] += frameT
          elif framePeriod == "T3":
            leftTime[2] += frameT
          # If the conf was low and sampling aligns, send to active learning pipeline
          if results[0].probs.top1conf.cpu().numpy() < 0.6 and frameCount % 5000 == 0:
            project.upload(left_frame)
        else:
          leftArr.append(0)

        # Right
        if results[1].probs.top1 == 0:
          # Calculate Object Being Interacted with
          if len(objArrR) > 1 and rC is not None:
            objR = calcDist(rC,objArrR)
            if objR[0][0] < objR[1][0]:
              objRightInt.append(objR[0][1])
            else:
              objRightInt.append(objR[0][1])
          rightArr.append(1)
          if framePeriod == "T1":
            rightTime[0] += frameT
          elif framePeriod == "T2":
            rightTime[1] += frameT
          elif framePeriod == "T3":
            rightTime[2] += frameT
          if results[1].probs.top1conf.cpu().numpy() < 0.6 and frameCount % 5000 == 0:
            project.upload(right_frame)
        else:
          rightArr.append(0)

    # Update Frame Count
    frameCount += 1

    # Stop if 10 minutes are processed
    if frameCount * frameT >= 10 * 60:
        break

  # Run Approach Detection
  rightAp = count_groups_of_ones(rightArr,3)
  leftAp = count_groups_of_ones(leftArr,3)

  # Run Per Object Count
  rightObj = count_obj_approach(objRightInt)
  leftObj = count_obj_approach(objLeftInt)

  # Calculate Average Time Per Approach
  rightApT = rightAp/frameT
  leftApT = leftAp/frameT

  return durationInSeconds, leftTime, rightTime, rightAp, leftAp, rightApT, leftApT, rightObj, leftObj, frames
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
mousePath = 'Models/yolov8s_mouse_detect_0.9.pt'
interactorPath = 'Models/interactM.pt'
yoloLocalizer = YOLO(localizerPath).to(device)
yoloMouse = YOLO(mousePath).to(device)
yoloInteractor = YOLO(interactorPath).to(device)

# Setup Batch Inference
video_extensions = ['.mp4', '.avi', '.MOV', '.mkv']
column = ["VideoName", "leftTime", "rightTime", "leftAp", "rightAp", "leftApT", "rightApT", "leftObj", "rightObj"]
dfTot = pd.DataFrame(columns=column)
vidIndex = 1

# Initialize Progress Bar
progress_text = f"Processing Video: {vidIndex}"
my_bar = st.progress(0, text=progress_text)

# Initialize Frame List
globalFrames = []
if start:
    for monthFolder in os.listdir(folder):
        path = os.path.join(folder, monthFolder)
        for file in os.listdir(path):
            if any(file.endswith(ext) for ext in video_extensions):
                video = os.path.join(path, file)
                totalDuration, leftTime, rightTime, rightAp, leftAp, rightApT, leftApT, rightObj, leftObj, frames = labelNOR(vidIndex, video, sampleRate, yoloInteractor, yoloMouse, yoloLocalizer, my_bar)
                data = {
                    "VideoName": [video],
                    "leftTime": [leftTime],
                    "rightTime": [rightTime],
                    "leftAp": [leftAp],
                    "rightAp": [rightAp],
                    "leftApT": [leftApT],
                    "rightApT": [rightApT],
                    "leftObj": [leftObj],
                    "rightObj": [rightObj],
                }
                df = pd.DataFrame(data)
                dfTot = pd.concat([dfTot, df], ignore_index=True)
                vidIndex += 1
                globalFrames.extend(frames)
                my_bar.empty()
    current_datetime = datetime.now()
    
    st.download_button(
        label="Download data as CSV",
        data=dfTot.to_csv(),
        file_name=f"{current_datetime}.csv",
        mime="text/csv",
    )

    # Convert frames to PIL Images
    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in globalFrames]

    # Use session state to keep track of the current image index
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    # Create three columns for layout
    left_col, center_col, right_col = st.columns([1, 3, 1])

    # Display the current image in the center column
    with center_col:
        if pil_images:
            st.image(pil_images[st.session_state.image_index], use_column_width=True)
        else:
            st.write("No frames to display.")

    # Create two columns for buttons within the center column
    button_col1, button_col2 = center_col.columns(2)

    # Back button
    with button_col1:
        if st.button("◀ Previous"):
            st.session_state.image_index = (st.session_state.image_index - 1) % len(pil_images)
            st.experimental_rerun()

    # Forward button
    with button_col2:
        if st.button("Next ▶"):
            st.session_state.image_index = (st.session_state.image_index + 1) % len(pil_images)
            st.experimental_rerun()
