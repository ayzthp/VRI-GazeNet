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


def calculate_gaze_strength(gaze_end, hand_center, nose_pos):
    """Calculate gaze strength/confidence based on angle precision and distance - Enhanced for desktop."""
    gaze_vector = np.array([gaze_end[0] - nose_pos[0], gaze_end[1] - nose_pos[1]])
    hand_vector = np.array([hand_center[0] - nose_pos[0], hand_center[1] - nose_pos[1]])
    
    # Calculate angle between gaze and hand direction
    angle = angle_between_vectors(gaze_vector, hand_vector)
    
    # Calculate distance to hand
    hand_distance = np.linalg.norm(hand_vector)
    
    # Enhanced gaze strength calculation for desktop - More sensitive
    # More forgiving angle thresholds for desktop use
    
    # Angle strength: More sensitive curve for desktop
    if angle < 5:
        angle_strength = 100  # Perfect alignment
    elif angle < 15:
        angle_strength = 95 - (angle - 5) * 1.5  # Very gradual decrease
    elif angle < 25:
        angle_strength = 80 - (angle - 15) * 2  # Moderate decrease
    elif angle < 40:
        angle_strength = 50 - (angle - 25) * 1.5  # Slower decrease
    elif angle < 60:
        angle_strength = 20 - (angle - 40) * 0.5  # Very slow decrease
    else:
        angle_strength = 0  # Too far off
    
    # Distance factor: More forgiving for desktop
    if hand_distance < 80:
        distance_factor = 0.9  # Too close but still acceptable
    elif hand_distance < 200:
        distance_factor = 1.0  # Optimal distance
    elif hand_distance < 400:
        distance_factor = 0.95  # Still very good
    elif hand_distance < 600:
        distance_factor = 0.85  # Acceptable
    else:
        distance_factor = 0.7  # Too far
    
    # Calculate total strength with enhanced normalization
    total_strength = angle_strength * distance_factor
    
    # Apply smoothing and normalization
    total_strength = max(0, min(100, total_strength))  # Clamp to 0-100
    
    return total_strength, angle


def is_gaze_focused_on_hand(gaze_end, hand_center, nose_pos, min_strength=50):
    """Check if gaze is strongly focused on the hand with high confidence - Lower threshold for desktop."""
    strength, angle = calculate_gaze_strength(gaze_end, hand_center, nose_pos)
    return strength >= min_strength, strength, angle


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


def check_sustained_gaze(gaze_strength_history, required_frames=15, min_strength=60):
    """Check if gaze has been sustained on hand for required number of frames."""
    if len(gaze_strength_history) < required_frames:
        return False, len(gaze_strength_history), required_frames
    
    # Check if the last required_frames have all been above min_strength
    recent_frames = gaze_strength_history[-required_frames:]
    sustained_count = sum(1 for strength in recent_frames if strength >= min_strength)
    
    return sustained_count >= required_frames, sustained_count, required_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description='Desktop VRI-GazeNet with connected camera (camera ID 8).')
    parser.add_argument('--gpu', dest='gpu_id', default='0', type=str)
    parser.add_argument('--snapshot', dest='snapshot', default='models/VRI.pkl', type=str)
    parser.add_argument('--cam', dest='cam_id', default=8, type=int, help='Camera ID for desktop (default: 8)')
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

    # Desktop Camera (ID 8)
    cap = cv2.VideoCapture(args.cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution for desktop
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera ID {args.cam_id}")
        print("Trying camera ID 0 as fallback...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open any camera")
            exit(1)
        else:
            print("✅ Using camera ID 0 as fallback")

    # Save video
    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('desktop_output.mp4', fourcc, 20.0, (1280,720))

    print("🎯 Desktop VRI-GazeNet System Started")
    print("=====================================")
    print(f"📷 Using camera ID: {args.cam_id}")
    print("🎮 Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save screenshot")
    print("- Show your LEFT hand for detection")
    print("")
    
    fps_cnt, fps_t0 = 0, time.time()
    fps = 0  # Initialize fps variable

    NOSE=1; LP=468; RP=473
    
    # Gaze strength smoothing for desktop
    gaze_strength_history = []
    smoothing_window = 5  # Average over 5 frames
    
    # Sustained gaze detection
    sustained_gaze_frames = 15  # Require 15 frames of sustained gaze
    sustained_gaze_min_strength = 60  # Minimum strength for sustained gaze

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("❌ Error: Could not read frame from camera")
            break
            
        # No flip for desktop camera (connected camera)
        h,w,_ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face landmarks
        face_results = face_mesh.process(rgb)
        # Process hand landmarks
        hand_results = hands.process(rgb)
        
        left_hand_detected = False
        hand_center = None
        
        # Detect left hand
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
                    is_focused, gaze_strength, gaze_angle = is_gaze_focused_on_hand(arrow_end, hand_center, nose_pt, min_strength=50)
                    
                    # Apply smoothing to gaze strength
                    gaze_strength_history.append(gaze_strength)
                    if len(gaze_strength_history) > smoothing_window:
                        gaze_strength_history.pop(0)
                    
                    # Calculate smoothed gaze strength
                    smoothed_strength = sum(gaze_strength_history) / len(gaze_strength_history)
                    
                    # Check for sustained gaze
                    is_sustained, sustained_count, required_frames = check_sustained_gaze(
                        gaze_strength_history, sustained_gaze_frames, sustained_gaze_min_strength
                    )
                    
                    # Always show gaze strength info (smoothed)
                    cv2.putText(frame, f"Gaze Strength: {smoothed_strength:.1f}%", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Angle: {gaze_angle:.1f}°", (10, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show sustained gaze progress
                    progress_text = f"Sustained: {sustained_count}/{required_frames}"
                    progress_color = (0, 255, 0) if is_sustained else (255, 255, 0)
                    cv2.putText(frame, progress_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, progress_color, 2)
                    
                    # Show stability indicator
                    if len(gaze_strength_history) >= smoothing_window:
                        stability = "STABLE" if max(gaze_strength_history) - min(gaze_strength_history) < 10 else "UNSTABLE"
                        stability_color = (0, 255, 0) if stability == "STABLE" else (0, 165, 255)
                        cv2.putText(frame, f"Status: {stability}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 2)
                    
                    # Only show focused indicators if sustained gaze is achieved
                    if is_sustained and smoothed_strength >= 60:
                        # Draw blue lines from pupils to hand only when truly focused and sustained
                        draw_pupil_to_hand_lines(left_pt, right_pt, hand_center, frame, color=(255, 0, 0), thickness=2)
                        
                        # Check if nose line bisects the pupil angle
                        bisects = does_nose_line_bisect_pupil_angle(nose_pt, arrow_end, left_pt, right_pt, hand_center)
                        
                        if bisects:
                            # Draw green bisecting line from nose to hand center
                            cv2.line(frame, nose_pt, hand_center, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(frame, "PERFECT ALIGNMENT!", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.putText(frame, "FOCUSED ON HAND", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, "SUSTAINED GAZE", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # Show that hand is detected but not sustained
                        if smoothed_strength >= 60:
                            focus_status = "BUILDING SUSTAINED GAZE"
                            color = (255, 255, 0)  # Yellow
                        elif smoothed_strength > 30:
                            focus_status = "WEAK FOCUS"
                            color = (0, 165, 255)  # Orange
                        else:
                            focus_status = "NOT FOCUSED"
                            color = (128, 128, 128)  # Gray
                        
                        cv2.putText(frame, focus_status, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Annotate gaze info
                cv2.putText(frame, f"Yaw: {yaw:.1f}° Pitch: {pitch:.1f}°", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
        else:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if not left_hand_detected:
            cv2.putText(frame, "Show LEFT hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Camera info
        cv2.putText(frame, f"Camera ID: {args.cam_id}", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, "Desktop Mode", (w-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # FPS
        fps_cnt +=1
        if time.time()-fps_t0>=1.0:
            fps = fps_cnt/(time.time()-fps_t0)
            fps_cnt, fps_t0 = 0, time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.imshow('Desktop VRI-GazeNet', frame)
        if args.save_video: out.write(frame)
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k==ord('s'):
            fn = f"desktop_frame_{int(time.time())}.jpg"
            cv2.imwrite(fn, frame)
            print(f"📸 Saved {fn}")

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    print("✅ Desktop VRI-GazeNet stopped.") 