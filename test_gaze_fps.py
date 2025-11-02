import cv2
import time
import numpy as np
from gaze_tracking import ScreenEngagementDetector
import statistics

print("=" * 60)
print("GAZE TRACKING FPS TEST (Client-Side Processing)")
print("=" * 60)

# Initialize detector
detector = ScreenEngagementDetector()

# Open webcam with better initialization
print("\nInitializing webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit(1)

# Wait for camera to warm up
print("Warming up camera...")
time.sleep(2)

# Capture several frames until we get a good one
print("Capturing calibration frame...")
for i in range(10):
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        print(f"✓ Got valid frame (attempt {i+1})")
        break
    time.sleep(0.1)
else:
    print("❌ Cannot read frame after 10 attempts")
    cap.release()
    exit(1)

# Display frame info
h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}")

# Do calibration
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = detector.face_mesh.process(frame_rgb)

if results.multi_face_landmarks:
    face_landmarks = results.multi_face_landmarks[0].landmark
    head_center, R_final, nose_points_3d = detector.compute_head_pose(face_landmarks, w, h)
    detector.calibrate(face_landmarks, head_center, R_final, nose_points_3d, w, h)
    print("✓ Calibrated successfully\n")
else:
    print("❌ No face detected for calibration")
    print("Make sure you're in front of the camera and well-lit!")
    cap.release()
    exit(1)

# Measure processing time for 100 frames
print("Measuring gaze tracking speed (100 frames)...")
print("Stay in front of the camera...\n")
frame_times = []
successful_frames = 0

for i in range(150):  # Try up to 150 to get 100 good frames
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    
    start = time.time()
    processed_frame, engaged, gaze_angle = detector.process_frame(frame)
    frame_time = (time.time() - start) * 1000  # Convert to ms
    
    frame_times.append(frame_time)
    successful_frames += 1
    
    if successful_frames % 20 == 0:
        print(f"  Processed {successful_frames}/100 frames...")
    
    if successful_frames >= 100:
        break

cap.release()

if len(frame_times) < 50:
    print(f"\n❌ Only got {len(frame_times)} frames. Need at least 50.")
    exit(1)

# Calculate statistics
mean_time = statistics.mean(frame_times)
std_time = statistics.stdev(frame_times)
p95_time = sorted(frame_times)[int(len(frame_times) * 0.95)]
fps = 1000 / mean_time

print("\n" + "=" * 60)
print("📊 RESULTS:")
print("=" * 60)
print(f"Frames analyzed:     {len(frame_times)}")
print(f"Mean frame time:     {mean_time:.1f} ms")
print(f"Std deviation:       {std_time:.1f} ms")
print(f"95th percentile:     {p95_time:.1f} ms")
print(f"FPS:                 {fps:.1f}")
print(f"Min:                 {min(frame_times):.1f} ms")
print(f"Max:                 {max(frame_times):.1f} ms")
print(f"Target (<100ms):     {'✅ PASS' if mean_time < 100 else '❌ FAIL'}")
print("=" * 60)

print("\n📋 FOR YOUR TABLE III:")
print(f"Gaze tracking/frame    {mean_time:.0f} ± {std_time:.0f}         {p95_time:.0f}            {'✓' if mean_time < 100 else '✗'} (<100ms)")