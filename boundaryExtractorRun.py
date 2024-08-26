import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch 
import glob
import os
import random
from tqdm import tqdm


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

def extract_large_bounding_box(image, yoloMouse, yoloLocalizer, padding=15):
    image = cv2.resize(image, (416,416))
    cropArr= []
    
    # Detect objects
    object_results = yoloLocalizer(image, verbose=False)
    object_boxes = []
    for obj in object_results:
        object_boxes.append(obj.boxes.xyxy.cpu().numpy())
    if not object_boxes:
        #print("No objects detected.")
        return None
    #print(f"Object bounding boxes: {object_boxes}")
    
    # Detect mouse
    mouse_results = yoloMouse(image, verbose=False)
    mice = []
    for mouse in mouse_results:
        mice.append(mouse.boxes.xyxy.cpu().numpy())
    for mouse in mice[0]:
        # If only one object, use it. Otherwise, find the closest.
        if len(object_boxes[0]) == 1:
            chosen_object_box = object_boxes[0]
            large_box = create_convex_hull_box(mouse, chosen_object_box[0])

        else:
            
            mouse_centroid = get_centroid(mouse)
            object_centroids = []
            for box in object_boxes[0]:
                object_centroids.append(get_centroid(box))
            distances = [np.linalg.norm(mouse_centroid - obj_centroid) for obj_centroid in object_centroids]
            #print(distances)
            if len(distances) > 1:
                chosen_object_box = object_boxes[0][np.argmin(distances)]
                large_box = create_convex_hull_box(mouse, chosen_object_box)

            else:
                return None

        #print(f"Chosen object bounding box: {chosen_object_box}")

        # Create convex hull bounding box

        # Add padding
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

        #print(f"Final bounding box: {large_box}")

        # Crop the image
        cropped_image = image[large_box[1]:large_box[3], large_box[0]:large_box[2]]
        cropArr.append(cropped_image)
    
    return cropArr

def visualize_boxes(image, boxes, colors):
    img_copy = image.copy()
    for box, color in zip(boxes, colors):
        cv2.rectangle(img_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    return img_copy

def save_cropped_images(base_save_dir, category,  crop):
    # Create the directory if it doesn't exist
    save_dir = os.path.join(base_save_dir, category)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate a random integer for the filename
    random_filename = f"{random.randint(1000, 999999)}.jpg"

    # Construct the full path to save the image
    save_path = os.path.join(save_dir, random_filename)

    # Save the cropped image
    crop = cv2.resize(crop, (256,256))
    cv2.imwrite(save_path, crop)
    #print(f"Cropped image saved to {save_path}")
    
# Main execution
if __name__ == "__main__":
    # Load YOLO models
    torch.cuda.set_device(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    localizerPath = 'app/Models/yolov9_objectLocalizer.pt'
    mousePath = 'app/Models/mouse_detection_yolov9c.pt'
    yoloLocalizer = YOLO(localizerPath).to(device)
    yoloMouse = YOLO(mousePath).to(device)
    print("[STATUS] YOLO Initialized")

    # Define the base directory
    base_dir = 'C:/Users/aalis/Downloads/MouseInteraction.v10i.folder/valid'
    base_save_dir = 'C:/Users/aalis/Downloads/MouseInteraction.v10i.folder/cropped_images'

    # Use glob to get all image paths in the interaction and noninteraction folders
    interaction_images = glob.glob(os.path.join(base_dir, 'Interaction', '*'))
    noninteraction_images = glob.glob(os.path.join(base_dir, 'NoInteraction', '*'))

    # Run Pipeline
    for image_path in tqdm(interaction_images, desc="Processing Interaction Images"):
        image = cv2.imread(image_path)
        if image is None:
            continue
        else:
            cropArr = extract_large_bounding_box(image, yoloMouse, yoloLocalizer)
            if cropArr is not None:
                for crop in cropArr:
                    # Save cropped image
                    save_cropped_images(base_save_dir, 'Interaction', crop)
            
    
    for image_path in tqdm(noninteraction_images, desc="Processing NoInteraction Images"):
        image = cv2.imread(image_path)
        if image is None:
            continue
        else:
            cropArr = extract_large_bounding_box(image, yoloMouse, yoloLocalizer)
            if cropArr is not None:
                for crop in cropArr:
                    # Save cropped image with a random filename in the noninteraction folder
                    save_cropped_images(base_save_dir, 'NoInteraction', crop)
            