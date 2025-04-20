import cv2
from mtcnn import MTCNN
import os

base_path = "/content/drive/My Drive/myDataset"
output_path = "/content/drive/My Drive/myDataset/extracted_faces"

# Define face detector
detector = MTCNN()

def process_images(input_dir, output_dir):
    """ Process all images in the given directory and save cropped faces """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all subfolders
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)

        if os.path.isdir(folder_path):  # Check if it's a folder
            print(f"Processing folder: {folder_path}")

            # Create corresponding output subfolder
            save_folder = os.path.join(output_dir, folder)
            os.makedirs(save_folder, exist_ok=True)

            # Process all images in the folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                if not os.path.isfile(image_path):  # Skip subfolders
                    continue

                try:
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    faces = detector.detect_faces(image)

                    if len(faces) == 0:
                        print(f"No faces detected in {image_name}")
                        continue

                    for i, face in enumerate(faces):
                        x, y, width, height = face['box']

                        # Add margin (30% extra space)
                        margin_x = int(0.3 * width)
                        margin_y = int(0.3 * height)

                        x1 = max(0, x - margin_x)
                        y1 = max(0, y - margin_y)
                        x2 = min(image.shape[1], x + width + margin_x)
                        y2 = min(image.shape[0], y + height + margin_y)

                        # Crop and save the face
                        face_crop = image[y1:y2, x1:x2]
                        face_filename = f"{os.path.splitext(image_name)[0]}-{i}.png"
                        save_path = os.path.join(save_folder, face_filename)
                        cv2.imwrite(save_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")

# Process real and fake images separately
process_images(os.path.join(base_path, "real"), os.path.join(output_path, "faces_real"))
process_images(os.path.join(base_path, "fake"), os.path.join(output_path, "faces_fake"))

print("Face extraction completed!")
