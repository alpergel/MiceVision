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
def processSideOld(object_model, frame,mouseSide, padding, num):
    objects = object_model(frame, verbose=False, max_det = 4)
    Boxes = []
    for obj in objects:
        Boxes.append(obj.boxes.xyxy.cpu().numpy())
    Boxes = sorted(Boxes[0], key=lambda o: get_centroid(o)[1])
    mouse_centroid = get_centroid(mouseSide[0].boxes.xyxy.cpu().numpy()[0])
    object_centroids = []
    for box in Boxes:
        object_centroids.append(get_centroid(box))
    distances = [np.linalg.norm(mouse_centroid - obj_centroid) for obj_centroid in object_centroids]
    if len(distances) > 1:
        chosen_object_box = Boxes[np.argmin(distances)]
        large_box = create_convex_hull_box(mouseSide[0].boxes.xyxy.cpu().numpy()[0], chosen_object_box)
        return [process_crop(frame, large_box, padding),num,np.argmin(distances)]
    return None

def match_mice_to_boxes(object_model, frame, mouseSides, padding):
    objects = object_model(frame, verbose=False, max_det=4)
    Boxes = []
    for obj in objects:
        Boxes.append(obj.boxes.xyxy.cpu().numpy())
    Mice = []
    for mous in mouseSides:
        Mice.append(mous.boxes.xyxy.cpu().numpy())
    Boxes = sorted(Boxes[0], key=lambda o: get_centroid(o)[1])  # Sort Boxes by Y
    Mice = sorted(Mice[0], key=lambda o: get_centroid(o)[0])   # Sort Mice by X
    if len(Mice) < 2 or len(Boxes) < 2:
        return None  # Ensure there are at least 2 mice and 2 boxes

    mouse_centroids = [get_centroid(mouse) for mouse in Mice]
    object_centroids = [get_centroid(box) for box in Boxes]
    
    distances = np.zeros((len(mouse_centroids), len(object_centroids)))
    for i, mouse_centroid in enumerate(mouse_centroids):
        for j, obj_centroid in enumerate(object_centroids):
            distances[i, j] = np.linalg.norm(mouse_centroid - obj_centroid)
    
    matched_boxes = []
    for i in range(len(mouse_centroids)):
        closest_box_index = np.argmin(distances[i])
        matched_boxes.append((Mice[i], Boxes[closest_box_index]))
        distances[:, closest_box_index] = np.inf  # Exclude this box for the next mouse
    #print("BOX MATCH DEBUG: ", matched_boxes)
    matched_boxes = sorted(matched_boxes, key=lambda x: x[0][0])  # Sort by the x-coordinate of the mouse centroid
    
    large_boxes = []

    for mouse, box in matched_boxes:
        box_centroid = get_centroid(box)
        print(box_centroid, frame.shape[0]//2)
        # Determine if the box is on the left or right side
        if box_centroid[0] < frame.shape[1] // 2:
            side = 0
        else:
            side = 1
        
        # Determine if the box is on the top or bottom
        if box_centroid[1] < frame.shape[0] // 2:
            position = 1
        else:
            position = 0
        if len(mouse) >= 4 and len(box) >= 4:  # Ensure both mouse and box have enough coordinates
            large_boxes.append((create_convex_hull_box(mouse, box),(side,position)))
    cropped_images = [(process_crop(frame, large_box, padding)[0], position) for (large_box, position) in large_boxes]

    return cropped_images


def run_interaction(frame, mouse_model, object_model, padding=10):
    # left_frame = cv2.resize(left_frame, (416,416))
    # right_frame = cv2.resize(right_frame, (416,416))
    frame = cv2.resize(frame, (416,416))
    # Detect Mice
    mouseB = mouse_model(frame, verbose=False, max_det = 2)
    
    # mouseL = mouse_model(left_frame, verbose=False,  max_det = 1)[0]
    # mouseR = mouse_model(right_frame, verbose=False,  max_det = 1)[0]
    
    if len(mouseB) > 0:
        # Detect objects
        #crops = processSide(object_model,frame, mouseB, padding, 0)
        crops = match_mice_to_boxes(object_model,frame,mouseB, padding)

        # leftCrop = processSide(object_model, left_frame, mouseL, padding, 0)
        # rightCrop = processSide(object_model, right_frame, mouseR, padding, 1)
        
        # # Add to CropARR
        # if leftCrop is not None:
        #     cropArr.append(leftCrop)
        # if rightCrop is not None:
        #     cropArr.append(rightCrop)
        
        return crops if crops else None
    else:
        return None