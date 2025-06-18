import argparse
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from model import VRI_GazeNet
import os

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time gaze estimation using webcam with VRI-GazeNet.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='models/VRI.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--save', dest='save_video', help='Save output video',
        action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load model
    model = VRI_GazeNet(num_bins=181)
    print('Loading snapshot...')
    saved_state_dict = torch.load(snapshot_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)
    model.cpu()
    model.eval()
    print('Model loaded successfully!')

    # Initialize webcam
    cap = cv2.VideoCapture(args.cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    # Setup video writer if saving
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if args.save_video:
        out = cv2.VideoWriter('gaze_webcam_output.avi', fourcc, 20.0, (640, 480))

    print("Starting real-time gaze detection...")
    print("Press 'q' to quit")
    
    # FPS calculation
    fps_count = 0
    fps_time = time.time()
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Use center portion of image as face region (simplified approach)
            # In a real application, you'd want to use actual face detection
            x_min = width // 4
            y_min = height // 4
            x_max = 3 * width // 4
            y_max = 3 * height // 4
            
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            # Crop and preprocess image
            img = frame[y_min:y_max, x_min:x_max]
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            img_tensor = transformations(im_pil)
            img_tensor = Variable(img_tensor).cpu()
            img_tensor = img_tensor.unsqueeze(0)
            
            # Gaze prediction
            gazes = model.angles(img_tensor)
            yaw, pitch = gazes[0]
            
            yaw_predicted = yaw * np.pi/180.0
            pitch_predicted = pitch * np.pi/180.0
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw gaze direction
            draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, 
                     (yaw_predicted, pitch_predicted), color=(0, 255, 255), 
                     scale=1.5, thickness=3, size=bbox_width, 
                     bbox=((x_min, y_min), (x_max, y_max)))
            
            # Add text information
            cv2.putText(frame, f"Yaw: {yaw:.1f}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}°", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Calculate and show FPS
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_count / (time.time() - fps_time)
                fps_count = 0
                fps_time = time.time()
                
            cv2.putText(frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save frame if requested
            if args.save_video:
                out.write(frame)
            
            # Display frame
            cv2.imshow('Real-time Gaze Detection', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # Cleanup
    cap.release()
    if args.save_video:
        out.release()
        print("Video saved as 'gaze_webcam_output.avi'")
    cv2.destroyAllWindows()
    print("Real-time gaze detection stopped.") 