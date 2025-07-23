import cv2
import os
import glob

# Path to your videos folder
videos_folder = 'videos'
# Path to save frames
frames_folder = 'frames'
os.makedirs(frames_folder, exist_ok=True)

# Extract every Nth frame
frame_interval = 10

# Get all video files in the videos folder
video_files = glob.glob(os.path.join(videos_folder, '*.mp4'))

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_num = 0
    saved_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{frame_num:05d}.jpg"
            cv2.imwrite(os.path.join(frames_folder, frame_filename), frame)
            saved_num += 1
        frame_num += 1
    cap.release()
    print(f"Extracted {saved_num} frames from {video_name}")