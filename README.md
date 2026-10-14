# Deepfake Protector AI

A face-focused deepfake image classification project built with Python, TensorFlow/Keras, OpenCV, and XceptionNet. The project explores identifying manipulated facial content using transfer learning and exposes an image-prediction endpoint through FastAPI.

> **Project status:** Academic project / research prototype. The reported evaluation scores below come from the project report and have not been independently reproduced as part of this README. Check the implementation notes before relying on predictions.

## Overview

Deepfakes can make synthetic or manipulated media appear authentic. This project builds a binary classifier intended to distinguish real facial images from deepfake facial images. Its data pipeline extracts frames from videos, detects and crops faces, prepares datasets, and trains an XceptionNet-based classifier. A FastAPI endpoint accepts an image for inference.

## Features

- **Video frame extraction:** Uses OpenCV to extract frames from real and fake video folders.
- **Face detection and cropping:** Uses MTCNN in the dataset-preparation pipeline to isolate face regions.
- **Transfer learning:** Uses an ImageNet-pretrained XceptionNet backbone with a custom binary-classification head.
- **Dataset preparation:** Organizes face images into training, validation, and test splits.
- **Image inference API:** Provides a FastAPI endpoint at `POST /predictImage` for uploading one image.

## How It Works

1. Collect real and fake videos from the selected datasets.
2. Extract frames from the videos with OpenCV.
3. Detect faces and save cropped face images for training.
4. Split the prepared images into training, validation, and test sets.
5. Train the classifier using the XceptionNet feature extractor and a custom classification head.
6. Load the trained model in the inference pipeline, crop a face from an uploaded image, resize it to `128 × 128`, normalize pixel values, and run prediction.

### Model Architecture

The training script builds a model with:

- **Base model:** XceptionNet pretrained on ImageNet, with the original classification top removed and the backbone frozen.
- **Global average pooling:** Converts feature maps into a feature vector.
- **Dense layer:** 512 units with ReLU activation.
- **Dropout:** Rate of `0.5` to help reduce overfitting.
- **Dense layer:** 128 units with ReLU activation.
- **Output layer:** One sigmoid unit for binary classification.

The model is compiled with binary cross-entropy and the Adam optimizer.

## Reported Evaluation

The project report records the following evaluation results:

| Metric | Reported score |
|---|---:|
| Accuracy | 98.57% |
| Precision | 99.39% |
| Recall | 99.29% |
| F1-score | 99.34% |

These are the values recorded in the project report; the README does not establish the dataset split, evaluation protocol, or independent reproducibility of these scores. Performance may vary with the dataset, preprocessing, and training configuration.

## Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow, Keras, XceptionNet
- **Image processing:** OpenCV, NumPy
- **Face detection:** MTCNN for dataset preprocessing; OpenCV Haar Cascade in the current API inference helper
- **API:** FastAPI
- **Development environment:** Google Colab was used for model development and training

## Datasets

The project report names these datasets:

- DeepfakeTIMIT
- FaceForensics
- FaceForensics++

The older repository README also mentions Google Deep Fake Detection (DFD). Confirm which datasets were included in the final training run before using that list as a reproducibility claim. Dataset files are not included in this repository and may have separate access terms.

## Repository Structure

```text
.
├── app.py                  # FastAPI application and prediction endpoint
├── features.py             # Image preprocessing and model inference helper
├── trained.py              # XceptionNet training script
├── New_VideoToImage.py     # Extract frames from videos
├── New_Mtcnn.py            # Detect and crop faces with MTCNN
├── New_PreparedDataset.py  # Prepare and split the face-image dataset
├── requirement.txt         # Python package list
└── LICENSE                 # MIT License
```

The inference helper expects the trained model at `models/best_model.h5`. Make sure that file is available locally before starting the API. The training script saves a checkpoint under the configured dataset/checkpoint directory; adjust the paths in the scripts for your environment.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/abhaysharma20233009/Deepfake-Protector-AI-Project-Sem4.git
cd Deepfake-Protector-AI-Project-Sem4
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirement.txt
pip install fastapi uvicorn python-multipart
```

The last command installs the API server and multipart-upload packages used by `app.py`, which are not currently listed in `requirement.txt`. TensorFlow/Keras compatibility can depend on the Python version, so use a compatible environment if installation fails.

### 4. Add the trained model

Place the trained model at:

```text
models/best_model.h5
```

Create the `models` directory if needed. The repository code loads this exact path in `features.py`. The model and datasets may need to be obtained from your training environment; they are not listed among the tracked root files.

### 5. Start the API

```bash
uvicorn app:app --reload
```

When the server starts, open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the interactive API documentation.

## API Usage

### `GET /`

Checks that the API is responding.

### `POST /predictImage`

Uploads a single image using the form field `image`.

Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predictImage" \
  -F "image=@sample.jpg"
```

The current endpoint returns a numeric `result` rather than an explicit `Real` / `Deepfake` label. A value of `-1` is used by the inference helper when it cannot detect a face. See the implementation notes below before interpreting other values.

## Training Pipeline

The scripts use Google Colab-style paths under `/content/drive/My Drive/myDataset/`. Before running them outside that environment, update the dataset and output paths to match your local setup. The expected data is arranged into real/fake categories before frames and face crops are prepared, and the training script expects `train`, `val`, and `test` directories under `split_dataset`.

The project report records 20 epochs and a batch size of 346, while the current `trained.py` configures a batch size of 32. Treat the script as configurable code and verify the final training settings used to produce the reported metrics.

## Implementation Notes / Limitations

- The dataset-preparation script uses MTCNN, but the current `features.py` inference path uses OpenCV's Haar Cascade face detector. These are different preprocessing paths and may affect predictions.
- The model file at `models/best_model.h5` must be supplied locally for the API to load successfully.
- `features.py` currently applies `np.argmax()` to the model's single-unit sigmoid output. For a one-unit sigmoid classifier, this does not provide a meaningful binary class decision. Use a threshold on the predicted probability and explicitly verify the training label mapping before interpreting predictions as real or fake.
- The API currently accepts an image, not a video. Video support would require extracting and processing frames at inference time.
- Results can be unreliable for blurry, low-resolution, occluded, or out-of-distribution faces. This prototype should not be treated as a definitive authenticity or forensic system.

## Future Improvements

- Evaluate on larger and more diverse datasets.
- Explore ensemble models and improved preprocessing.
- Add temporal consistency analysis for video-level detection.
- Improve inference performance and deploy a user-facing web or mobile interface.
- Explore audio-based deepfake detection.

## Team

- **Abhay Sharma** — Project Lead
- Anshu Gupta
- Abdul Samad Ansari
- Aman Poddar
- Aditya Kumar

## License

This project is distributed under the [MIT License](LICENSE).
