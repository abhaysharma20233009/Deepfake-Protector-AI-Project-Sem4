import os
import cv2
import shutil
import json
import numpy as np
import splitfolders
from mtcnn import MTCNN

# Define paths
base_path = '/content/drive/My Drive/myDataset/'
extracted_faces_path = os.path.join(base_path, 'extracted_faces')
faces_fake_path = os.path.join(extracted_faces_path, 'faces_fake')
faces_real_path = os.path.join(extracted_faces_path, 'faces_real')

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# # Create necessary directories
# create_dir(extracted_faces_path)
# create_dir(faces_fake_path)
# create_dir(faces_real_path)

def get_all_images(directory):
    all_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    return all_images

# def extract_faces(input_path, output_path):
#     detector = MTCNN()
#     for subdir in os.listdir(input_path):  # Loop through subdirectories
#         subdir_path = os.path.join(input_path, subdir)
#         if os.path.isdir(subdir_path):
#             for image_file in os.listdir(subdir_path):
#                 image_path = os.path.join(subdir_path, image_file)
#                 image = cv2.imread(image_path)
#                 if image is None:
#                     continue
#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 results = detector.detect_faces(image_rgb)
#                 count = 0
#                 for result in results:
#                     x, y, width, height = result['box']
#                     x1, y1 = max(0, x), max(0, y)
#                     x2, y2 = min(image.shape[1], x + width), min(image.shape[0], y + height)
#                     cropped_face = image_rgb[y1:y2, x1:x2]
#                     face_filename = f"{subdir}_{image_file.split('.')[0]}_{count}.png"
#                     cv2.imwrite(os.path.join(output_path, face_filename), cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     count += 1

# Extract faces from fake and real directories
# extract_faces(os.path.join(base_path, 'fake'), faces_fake_path)
# extract_faces(os.path.join(base_path, 'real'), faces_real_path)

# Balance fake faces to match real faces count
real_faces = get_all_images(faces_real_path)
fake_faces = get_all_images(faces_fake_path)

if len(fake_faces) > len(real_faces):
    sampled_fake_faces = np.random.choice(fake_faces, len(real_faces), replace=False)
    for f in fake_faces:
        if f not in sampled_fake_faces:
            os.remove(f)

print("Face extraction and balancing completed.")

# Prepare dataset for training
prepared_dataset_path = os.path.join(base_path, 'prepared_dataset')
create_dir(prepared_dataset_path)
shutil.copytree(faces_real_path, os.path.join(prepared_dataset_path, 'real'), dirs_exist_ok=True)
shutil.copytree(faces_fake_path, os.path.join(prepared_dataset_path, 'fake'), dirs_exist_ok=True)

# Split dataset into train, val, and test
splitfolders.ratio(prepared_dataset_path, output=os.path.join(base_path, 'split_dataset'), seed=1377, ratio=(.8, .1, .1))
print("Dataset split into Train/Val/Test.")

