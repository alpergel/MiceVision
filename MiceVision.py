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


def labelNOR(video, sampleRate, yoloInteractor, yoloMouse, yoloLocalizer):
    # Create video capture object
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None  # Return None instead of exit() for better error handling

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    totalNoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameT = totalNoFrames / (fps * totalNoFrames)  # Time per frame

    # Initialize counters and arrays
    leftTime = np.zeros(3, dtype=np.float32)
    rightTime = np.zeros(3, dtype=np.float32)
    leftArr = []
    rightArr = []
    objLeftInt = []
    objRightInt = []
    frames = []
    splitframes = []

    # Pre-compute time periods
    T1, T2, T3 = 150, 300, 600
    time_periods = np.array([T1, T2, T3])

    # Initialize object arrays
    objArrL = []
    objArrR = []
    firstObj = False

    # Use deques for efficient append and pop operations
    leftArr = deque()
    rightArr = deque()

    for frameCount in range(0, min(int(10 * 60 * fps), totalNoFrames), sampleRate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameCount)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert Frames to Greyscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update progress
        my_bar.progress(round((frameCount/totalNoFrames)*100), text=progress_text)

        # Split frame
        midpoint = w // 2
        left_frame = frame[:, :midpoint]
        right_frame = frame[:, midpoint:]
        left_frame_grey = gray_frame[:, :midpoint]
        right_frame_grey = gray_frame[:, midpoint:]

        # Append frame
        frames.append(gray_frame)
        splitframes.append(left_frame_grey)
        splitframes.append(right_frame_grey)


        # Detect time period
        frametime = frameCount * frameT
        framePeriod = np.searchsorted(time_periods, frametime)

        # Check for mouse presence and run YOLO
        mouseCheck = yoloMouse([left_frame, right_frame], verbose=False)
        
        if len(mouseCheck[0].boxes) > 0 and len(mouseCheck[1].boxes) > 0:
            if not firstObj:
                objects = yoloLocalizer([left_frame, right_frame], verbose=False, conf=0.5)
                objArrL = [centroid(obj) for obj in objects[0] if centroid(obj) is not None]
                objArrR = [centroid(obj) for obj in objects[1] if centroid(obj) is not None]
                firstObj = True

            lC = centroid(mouseCheck[0])
            rC = centroid(mouseCheck[1])
            st.image(left_frame_grey)
            results = yoloInteractor([left_frame_grey, right_frame_grey], verbose=True)
            
            # Left interaction
            if results[0].probs.top1 == 0:
                if len(objArrL) > 1 and lC is not None:
                    objL = min(objArrL, key=lambda x: np.linalg.norm(x - lC) if x is not None and lC is not None else np.inf)
                    objLeftInt.append(objL[0])
                leftArr.append(1)
                leftTime[framePeriod] += frameT

                #left_buffer_count = buffer_frames
            else:
                leftArr.append(0)
            
            # Right interaction
            if results[1].probs.top1 == 0:
                if len(objArrR) > 1 and rC is not None:
                    objR = min(objArrR, key=lambda x: np.linalg.norm(x - rC) if x is not None and rC is not None else np.inf)
                    objRightInt.append(objR[0])
                rightArr.append(1)
                rightTime[framePeriod] += frameT

                #right_buffer_count = buffer_frames
            else:
                rightArr.append(0)
        else:
            leftArr.append(2)
            rightArr.append(2)
        # Update buffer counts
        #left_buffer_count = max(0, left_buffer_count - 1)
        #right_buffer_count = max(0, right_buffer_count - 1)

    cap.release()

    # Compute final statistics
    leftArr = np.array(leftArr)
    print(leftArr)
    leftAp = np.sum(leftArr)
    rightArr = np.array(rightArr)
    print(rightArr)
    rightAp = np.sum(rightArr)

    rightObj = len(set(objRightInt))
    leftObj = len(set(objLeftInt))

    rightApT = rightAp * frameT
    leftApT = leftAp * frameT

    return leftTime, rightTime, rightAp, leftAp, rightApT, leftApT, rightObj, leftObj, frames, splitframes, leftArr, rightArr

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
interactorPath = 'Models/yolov8xl_grey.pt'
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
                leftTime, rightTime, rightAp, leftAp, rightApT, leftApT, rightObj, leftObj, frames, splitframes, leftArr, rightArr = labelNOR(video, sampleRate, yoloInteractor, yoloMouse, yoloLocalizer)
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
                st.session_state.processed_data['dfTot'] = pd.concat([st.session_state.processed_data['dfTot'], df], ignore_index=True)
                vidIndex += 1
                st.session_state.processed_data['globalFrames'].extend(splitframes)
                st.session_state.processed_data['all_left_arr'].extend(leftArr)
                st.session_state.processed_data['all_right_arr'].extend(rightArr)
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
            current_left = st.session_state.processed_data['all_left_arr'][st.session_state.processed_data['image_index']]
            current_right = st.session_state.processed_data['all_right_arr'][st.session_state.processed_data['image_index']]
            
            # Create a new image with text
            img_with_text = Image.new('RGB', (current_image.width, current_image.height + 30), color='white')
            img_with_text.paste(current_image, (0, 30))
            
            # Add text to the image
            draw = ImageDraw.Draw(img_with_text)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            text = f"Left: {current_left}, Right: {current_right}"
            draw.text((10, 5), text, font=font, fill='black')
            img_with_text = img_with_text.resize((720,720))
            st.image(img_with_text, use_column_width=True)
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
        if st.button("Incorrect", use_container_width=True):
            FRAME_DIR = "incorrect"
            if not os.path.exists(FRAME_DIR):
                os.makedirs(FRAME_DIR)
            img = np.array(pil_images[st.session_state.processed_data['image_index']])
            h, w, c = img.shape
            rand = random.randint(0,1000000)
            rightCrop = img[0:h, w//2:w]
            leftCrop = img[0:h, 0:w//2]
            filePathR = f"{FRAME_DIR}/frameRight{rand}.jpg"
            filePathL = f"{FRAME_DIR}/frameLeft{rand}.jpg"
            iio.imwrite(filePathR, rightCrop)
            iio.imwrite(filePathL, leftCrop)
            st.session_state.processed_data['incorrect'].append(filePathR)
            st.session_state.processed_data['incorrect'].append(filePathL)
            if (st.session_state.processed_data['image_index'] + 1) % len(pil_images) <= len(pil_images):
                st.session_state.processed_data['image_index'] = (st.session_state.processed_data['image_index'] + 1) % len(pil_images)
            else:
                st.session_state.processed_data['image_index'] = 0
            st.success(f"Image saved!", icon="✅")
        current_datetime = datetime.now()
        st.download_button(
            label="Download data as CSV",
            data=dfTot.to_csv(),
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
        


