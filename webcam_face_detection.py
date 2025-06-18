#!/usr/bin/env python3
"""
Real-time Gaze Detection with Posture Alignment
Using VRI_GazeNet model and MediaPipe FaceMesh for exact pupil & nose-tip landmarks
Implements alignment scoring between pupils-to-gaze vector
"""

import argparse
import numpy as np
import cv2
import time
import mediapipe as mp
import math

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn

from PIL import Image

from utils import select_device
from model import VRI_GazeNet
import os


def draw_gaze_from_nose(nose_x, nose_y, frame, yaw_pitch, length=150, thickness=3, color=(0, 0, 255)):
    """Draw a gaze arrow from the nose position and return end point."""
    yaw, pitch = yaw_pitch
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    end_x = int(nose_x + dx)
    end_y = int(nose_y + dy)
    cv2.arrowedLine(frame, (int(nose_x), int(nose_y)), (end_x, end_y),
                    color, thickness, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(frame, (int(nose_x), int(nose_y)), 4, color, -1)
    return (end_x, end_y)


def draw_pupil_lines(left_pupil, right_pupil, arrow_end, frame, color=(0, 255, 0), thickness=2):
    """Draw lines from both pupils to arrow end and mark pupils."""
    cv2.line(frame, left_pupil, arrow_end, color, thickness, cv2.LINE_AA)
    cv2.line(frame, right_pupil, arrow_end, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, left_pupil, 3, color, -1)
    cv2.circle(frame, right_pupil, 3, color, -1)


def draw_pupil_to_hand_lines(left_pupil, right_pupil, hand_center, frame, color=(255, 0, 0), thickness=2):
    """Draw blue lines from both pupils to hand center."""
    cv2.line(frame, left_pupil, hand_center, color, thickness, cv2.LINE_AA)
    cv2.line(frame, right_pupil, hand_center, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, left_pupil, 3, color, -1)
    cv2.circle(frame, right_pupil, 3, color, -1)


def angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees."""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 0
    cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def is_gaze_towards_hand(gaze_end, hand_center, nose_pos, threshold=100):
    """Check if gaze direction is towards the hand."""
    gaze_vector = np.array([gaze_end[0] - nose_pos[0], gaze_end[1] - nose_pos[1]])
    hand_vector = np.array([hand_center[0] - nose_pos[0], hand_center[1] - nose_pos[1]])
    
    # Calculate angle between gaze and hand direction
    angle = angle_between_vectors(gaze_vector, hand_vector)
    return angle < 45  # Within 45 degrees


def does_nose_line_bisect_pupil_angle(nose_pos, gaze_end, left_pupil, right_pupil, hand_center, tolerance=15):
    """Check if the nose-to-gaze line bisects the angle between pupil-to-hand lines."""
    # Vectors from pupils to hand
    left_to_hand = np.array([hand_center[0] - left_pupil[0], hand_center[1] - left_pupil[1]])
    right_to_hand = np.array([hand_center[0] - right_pupil[0], hand_center[1] - right_pupil[1]])
    
    # Nose to gaze vector
    nose_to_gaze = np.array([gaze_end[0] - nose_pos[0], gaze_end[1] - nose_pos[1]])
    
    # Calculate angles
    angle_left = angle_between_vectors(nose_to_gaze, left_to_hand)
    angle_right = angle_between_vectors(nose_to_gaze, right_to_hand)
    
    # Check if nose line bisects (angles should be roughly equal)
    return abs(angle_left - angle_right) < tolerance


def angle_between(u, v):
    """Return angle in degrees between two 2D vectors."""
    dot = u[0]*v[0] + u[1]*v[1]
    nu = math.hypot(u[0], u[1])
    nv = math.hypot(v[0], v[1])
    if nu*nv == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot/(nu*nv)))
    return np.degrees(np.arccos(cos_theta))


def calculate_gaze_strength(gaze_end, hand_center, nose_pos):
    """Calculate gaze strength/confidence based on angle precision and distance."""
    gaze_vector = np.array([gaze_end[0] - nose_pos[0], gaze_end[1] - nose_pos[1]])
    hand_vector = np.array([hand_center[0] - nose_pos[0], hand_center[1] - nose_pos[1]])
    
    # Calculate angle between gaze and hand direction
    angle = angle_between_vectors(gaze_vector, hand_vector)
    
    # Calculate distance to hand
    hand_distance = np.linalg.norm(hand_vector)
    
    # Gaze strength: higher when angle is smaller and hand is at reasonable distance
    # Strength decreases exponentially with angle
    angle_strength = max(0, 100 * np.exp(-angle / 15))  # Strong when angle < 15°
    
    # Distance factor (optimal at medium distances)
    distance_factor = min(1.0, max(0.5, 200 / max(hand_distance, 50)))
    
    total_strength = angle_strength * distance_factor
    return total_strength, angle


def is_gaze_focused_on_hand(gaze_end, hand_center, nose_pos, min_strength=60):
    """Check if gaze is strongly focused on the hand with high confidence."""
    strength, angle = calculate_gaze_strength(gaze_end, hand_center, nose_pos)
    return strength >= min_strength, strength, angle


def parse_args():
    parser = argparse.ArgumentParser(
        description='Real-time gaze estimation with hand detection and bisecting line analysis.')
    parser.add_argument('--gpu', dest='gpu_id', default='0', type=str)
    parser.add_argument('--snapshot', dest='snapshot', default='models/VRI.pkl', type=str)
    parser.add_argument('--cam', dest='cam_id', default=0, type=int)
    parser.add_argument('--save', dest='save_video', action='store_true')
    parser.add_argument('--arrow-length', dest='arrow_length', default=150, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True

    # Model setup
    model = VRI_GazeNet(num_bins=181)
    state = torch.load(args.snapshot, map_location='cpu')
    model.load_state_dict(state)
    model.cpu().eval()

    # Device
    select_device(args.gpu_id, batch_size=1)

    # Transforms
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Hand detection (only for left hand)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Webcam
    cap = cv2.VideoCapture(args.cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Save video
    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

    print("Press 'q' to quit, 's' to save frame")
    print("Show your LEFT hand to the camera for detection")
    fps_cnt, fps_t0 = 0, time.time()
    fps = 0  # Initialize fps variable

    NOSE=1; LP=468; RP=473

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face landmarks
        face_results = face_mesh.process(rgb)
        # Process hand landmarks
        hand_results = hands.process(rgb)
        
        left_hand_detected = False
        hand_center = None
        
        # Detect left hand (appears on right side due to camera flip)
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                # Check if it's the left hand
                if handedness.classification[0].label == 'Left':
                    left_hand_detected = True
                    
                    # Get hand bounding box
                    hand_x = [landmark.x * w for landmark in hand_landmarks.landmark]
                    hand_y = [landmark.y * h for landmark in hand_landmarks.landmark]
                    
                    x1, y1 = int(min(hand_x)), int(min(hand_y))
                    x2, y2 = int(max(hand_x)), int(max(hand_y))
                    
                    # Draw hand bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "LEFT HAND", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Calculate hand center
                    hand_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    cv2.circle(frame, hand_center, 5, (0, 255, 255), -1)
                    break

        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark
            nose = lm[NOSE]
            lp = lm[LP]; rp = lm[RP]
            nose_pt = (int(nose.x*w), int(nose.y*h))
            left_pt = (int(lp.x*w), int(lp.y*h))
            right_pt = (int(rp.x*w), int(rp.y*h))

            # Crop & predict gaze
            xs = [l.x*w for l in lm]; ys = [l.y*h for l in lm]
            x1,y1,x2,y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            pad=20
            x1,y1 = max(0,x1-pad), max(0,y1-pad)
            x2,y2 = min(w,x2+pad), min(h,y2+pad)
            face = frame[y1:y2, x1:x2]
            if face.size>0:
                face = cv2.resize(face,(224,224)); face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(face); t = tfm(pil).unsqueeze(0)
                with torch.no_grad():
                    gazes = model.angles(t)
                    yaw, pitch = gazes[0]
                yaw_r = yaw*np.pi/180; pitch_r = pitch*np.pi/180

                # Draw default gaze arrow (red)
                arrow_end = draw_gaze_from_nose(
                    nose_pt[0], nose_pt[1], frame,
                    (yaw_r,pitch_r), length=args.arrow_length,
                    thickness=3, color=(0,0,255)
                )
                
                # Face bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Hand-related analysis
                if left_hand_detected and hand_center:
                    # Check if gaze is strongly focused on hand
                    is_focused, gaze_strength, gaze_angle = is_gaze_focused_on_hand(arrow_end, hand_center, nose_pt, min_strength=60)
                    
                    # Always show gaze strength info
                    cv2.putText(frame, f"Gaze Strength: {gaze_strength:.1f}%", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Angle: {gaze_angle:.1f}°", (10, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if is_focused:
                        # Draw blue lines from pupils to hand only when truly focused
                        draw_pupil_to_hand_lines(left_pt, right_pt, hand_center, frame, color=(255, 0, 0), thickness=2)
                        
                        # Check if nose line bisects the pupil angle
                        bisects = does_nose_line_bisect_pupil_angle(nose_pt, arrow_end, left_pt, right_pt, hand_center)
                        
                        if bisects:
                            # Draw green bisecting line from nose to hand center
                            cv2.line(frame, nose_pt, hand_center, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(frame, "PERFECT ALIGNMENT!", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.putText(frame, "FOCUSED ON HAND", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        # Show that hand is detected but not focused
                        focus_status = "WEAK FOCUS" if gaze_strength > 30 else "NOT FOCUSED"
                        color = (0, 165, 255) if gaze_strength > 30 else (128, 128, 128)
                        cv2.putText(frame, focus_status, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Annotate gaze info
                cv2.putText(frame, f"Yaw: {yaw:.1f}° Pitch: {pitch:.1f}°", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
        else:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if not left_hand_detected:
            cv2.putText(frame, "Show LEFT hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # FPS
        fps_cnt +=1
        if time.time()-fps_t0>=1.0:
            fps = fps_cnt/(time.time()-fps_t0)
            fps_cnt, fps_t0 = 0, time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.imshow('Gaze-Hand Analysis', frame)
        if args.save_video: out.write(frame)
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k==ord('s'):
            fn = f"frame_{int(time.time())}.jpg"
            cv2.imwrite(fn, frame)
            print(f"Saved {fn}")

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
