import cv2
import numpy as np
import os
import csv
from ultralytics import YOLO

video_path = 'videos/sai_vs_ginting.mp4'  # Set your video path here

def load_trained_model():
    possible_paths = [
        'best.pt',
        'runs/detect/train/weights/best.pt',
        'runs/detect/train2/weights/best.pt',
        'runs/detect/train3/weights/best.pt'
    ]
    for model_path in possible_paths:
        if os.path.exists(model_path):
            print(f"Found model at: {model_path}")
            return YOLO(model_path)
    print("Model not found. Please ensure you have a trained model file (best.pt)")
    print("Checked paths:", possible_paths)
    return None

def assign_player_ids(player_boxes, frame_height):
    if len(player_boxes) != 2:
        return {}
    centers = [(i, (box[1] + box[3]) / 2) for i, box in enumerate(player_boxes)]
    sorted_players = sorted(centers, key=lambda x: x[1], reverse=True)
    mapping = {sorted_players[0][0]: 'Player 1', sorted_players[1][0]: 'Player 2'}
    return mapping

def process_video_with_overlay(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Frame dimensions: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")
    positions = []
    frame_count = 0
    hit_count = 0
    hit_frames = []
    net_y = frame_height // 2
    last_side = None
    miss_count = 0
    miss_threshold = 3
    rally_end_frame = None
    rally_winner = None
    rally_winner_display_counter = 0
    RALLY_WINNER_DISPLAY_FRAMES = int(fps * 5)
    last_shuttle_side = None
    rally_start_frame = 0
    rally_results = []
    GROUND_Y_THRESHOLD = frame_height - 100
    OUT_OF_COURT_MARGIN = 50
    rally_winner_to_display = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        shuttle_detected = False
        shuttle_x = None
        shuttle_y = None
        player_boxes = []
        player_confs = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    if class_id == 2:
                        shuttle_x = (x1 + x2) / 2
                        shuttle_y = (y1 + y2) / 2
                        shuttle_detected = True
                        cv2.circle(frame, (int(shuttle_x), int(shuttle_y)), 18, (0, 255, 255), 3)
                    elif class_id in [0, 1]:
                        player_boxes.append([x1, y1, x2, y2])
                        player_confs.append(confidence)
        player_id_map = assign_player_ids(player_boxes, frame_height)
        for i, box in enumerate(player_boxes):
            x1, y1, x2, y2 = map(int, box)
            label = player_id_map.get(i, f"Player ?")
            conf = player_confs[i]
            color = (255, 0, 0) if label == 'Player 1' else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        rally_end = False
        if shuttle_detected:
            positions.append((frame_count, shuttle_x, shuttle_y))
            miss_count = 0
            # Out of court
            if (shuttle_x is not None and (
                shuttle_x < -OUT_OF_COURT_MARGIN or shuttle_x > frame_width + OUT_OF_COURT_MARGIN or
                shuttle_y < -OUT_OF_COURT_MARGIN or shuttle_y > frame_height + OUT_OF_COURT_MARGIN)):
                rally_end = True
            # Hits ground
            elif shuttle_y is not None and shuttle_y > GROUND_Y_THRESHOLD:
                rally_end = True
        else:
            positions.append((frame_count, None, None))
            miss_count += 1
            if miss_count >= miss_threshold:
                rally_end = True
        if rally_end and rally_end_frame is None:
            rally_end_frame = frame_count
            # Predict rally winner
            if last_shuttle_side is not None:
                if last_shuttle_side == 0:
                    rally_winner = 'Player 2'
                else:
                    rally_winner = 'Player 1'
                rally_winner_to_display = rally_winner
                rally_winner_display_counter = RALLY_WINNER_DISPLAY_FRAMES
                print(f"[RALLY END] Frame: {rally_end_frame}, Winner: {rally_winner}, Last shuttle side: {last_shuttle_side}")
                rally_results.append([rally_start_frame, rally_end_frame, rally_winner])
            rally_start_frame = frame_count + 1
        if shuttle_y is not None:
            side = 1 if shuttle_y > net_y else 0
            if last_side is not None and side != last_side:
                hit_count += 1
                hit_frames.append(frame_count)
            last_side = side
            last_shuttle_side = side
        cv2.putText(frame, f"Hits: {hit_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(frame, f"Frame: {frame_count}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if frame_count in hit_frames:
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 255, 0), 20)

        # --- Real-time stats overlay ---
        # current_rallies = len(rally_results) # Removed as per edit hint
        # current_hits = hit_count # Removed as per edit hint
        # current_duration = frame_count / fps if fps else 0 # Removed as per edit hint
        # if rally_winner_to_display and rally_winner_display_counter > 0: # Removed as per edit hint
        #     winner_text = f"Last rally winner: {rally_winner_to_display}" # Removed as per edit hint
        # else: # Removed as per edit hint
        #     winner_text = "Last rally winner: -" # Removed as per edit hint
        # stats_lines = [ # Removed as per edit hint
        #     f"Total rallies: {current_rallies}", # Removed as per edit hint
        #     f"Total hits: {current_hits}", # Removed as per edit hint
        #     f"Duration: {current_duration:.1f} sec", # Removed as per edit hint
        #     winner_text # Removed as per edit hint
        # ] # Removed as per edit hint
        # y0 = 60 # Removed as per edit hint
        # for i, line in enumerate(stats_lines): # Removed as per edit hint
        #     y = y0 + i * 60 # Removed as per edit hint
        #     cv2.putText(frame, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4) # Removed as per edit hint
        # Show winner overlay in center if just ended # Removed as per edit hint
        if rally_winner_to_display and rally_winner_display_counter > 0:
            text = f"{rally_winner_to_display} wins the rally!"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 12)
            x = (frame_width - tw) // 2
            y = frame_height // 2
            cv2.rectangle(frame, (x-40, y-th-40), (x+tw+40, y+40), (0,0,0), -1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0,255,0), 12)
            rally_winner_display_counter -= 1
            if rally_winner_display_counter == 0:
                rally_winner_to_display = None
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    cap.release()
    print(f"Annotated video saved to: {output_path}")
    # --- FORCE FINAL RALLY END AT END OF VIDEO ---
    if (not rally_results or rally_results[-1][1] != frame_count - 1) and last_shuttle_side is not None:
        rally_end_frame = frame_count - 1
        if last_shuttle_side == 0:
            rally_winner = 'Player 2'
        else:
            rally_winner = 'Player 1'
        print(f"[FORCED FINAL RALLY END] Frame: {rally_end_frame}, Winner: {rally_winner}, Last shuttle side: {last_shuttle_side}")
        rally_results.append([rally_start_frame, rally_end_frame, rally_winner])

    # Save rally results to CSV
    csv_path = output_path.replace('.mp4', '_rally_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rally_start_frame', 'rally_end_frame', 'winner'])
        writer.writerows(rally_results)
    print(f"Rally results saved to: {csv_path}")

    # --- Generate and print a natural language summary ---
    total_rallies = len(rally_results)
    total_hits = hit_count
    total_frames = frame_count
    duration_sec = total_frames / fps if fps else 0
    if rally_results and rally_results[-1][2]:
        last_winner = rally_results[-1][2]
    else:
        last_winner = 'Unknown'
    summary = f"\n--- Rally Analysis Summary ---\n"
    summary += f"Video: {os.path.basename(video_path)}\n"
    summary += f"Total rallies detected: {total_rallies}\n"
    summary += f"Total hits in video: {total_hits}\n"
    summary += f"Total duration: {duration_sec:.1f} seconds\n"
    if total_rallies > 0:
        summary += f"Last rally winner: {last_winner}\n"
    else:
        summary += "No rally winner detected.\n"
    summary += "-----------------------------\n"
    print(summary)
    # Optionally, save summary to a text file
    with open(output_path.replace('.mp4', '_summary.txt'), 'w') as f:
        f.write(summary)

    # --- Overlay summary stats on last N black frames ---
    SUMMARY_DISPLAY_FRAMES = int(fps * 3)  # Show for 3 seconds
    summary_lines = [
        f"Total rallies: {total_rallies}",
        f"Total hits: {total_hits}",
        f"Duration: {duration_sec:.1f} sec",
        f"Last rally winner: {last_winner}"
    ]
    for _ in range(SUMMARY_DISPLAY_FRAMES):
        summary_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        y0 = frame_height // 3
        for i, line in enumerate(summary_lines):
            y = y0 + i * 80
            cv2.putText(summary_frame, line, (100, y), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), 6)
        out.write(summary_frame)
    out.release()
    print(f"Final video with stats overlay saved to: {output_path}")


def main():
    model = load_trained_model()
    if model is None:
        return
    output_folder = '/Users/saipraveen/Documents/badminton_analysis/yolo_overlay_videos/new_outputs'
    os.makedirs(output_folder, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_folder, f"{video_name}_hits_overlay.mp4")
    print(f"\n{'='*50}")
    print(f"Processing: {video_path}")
    print(f"{'='*50}")
    process_video_with_overlay(model, video_path, output_path)

if __name__ == "__main__":
    main() 