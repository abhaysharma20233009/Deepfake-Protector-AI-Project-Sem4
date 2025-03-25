import json
import os
import cv2
import math

base_path = '/content/drive/My Drive/DeepfakeTIMIT'  # Adjust if needed

def get_filename_only(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# Load metadata.json
metadata_path = os.path.join(base_path, 'metadata.json')
if not os.path.exists(metadata_path):
    print("Error: metadata.json not found!")
    exit()

with open(metadata_path) as metadata_json:
    metadata = json.load(metadata_json)

print(f"Total videos in metadata: {len(metadata)}")
print("Metadata Keys: ", list(metadata.keys())[:5])  # Print first 5 filenames for debugging

video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

for filename in metadata.keys():
    if not filename.lower().endswith(tuple(video_extensions)):
        continue  # Skip non-video files

    video_file = os.path.join(base_path, metadata[filename]["path"])  # ✅ Correct path extraction

    if not os.path.exists(video_file):
        print(f"Error: File {video_file} not found!")
        continue

    print(f"Processing {video_file}...")

    tmp_path = os.path.join(base_path, get_filename_only(filename))
    os.makedirs(tmp_path, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open {video_file}")
        continue

    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    count = 0

    while cap.isOpened():
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame number
        ret, frame = cap.read()

        if not ret:
            print(f"Error: No frames read from {filename}")
            break

        if frame_id % math.floor(frame_rate) == 0:
            print(f"Extracting frame {frame_id}")

            # Resize logic
            width, height = frame.shape[1], frame.shape[0]
            if width < 300:
                scale_ratio = 2
            elif width > 1900:
                scale_ratio = 0.33
            elif width > 1000:
                scale_ratio = 0.5
            else:
                scale_ratio = 1

            new_width = int(width * scale_ratio)
            new_height = int(height * scale_ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Save frame
            new_filename = os.path.join(tmp_path, f"{get_filename_only(filename)}-{count:03d}.png")
            success = cv2.imwrite(new_filename, resized_frame)

            if success:
                print(f"Saved Image: {new_filename}")
            else:
                print(f"Error: Could not save image {new_filename}")

            count += 1

    cap.release()
    print(f"Done processing {filename}!")