# Deepfake-Protector-AI-Project-Sem4
# Description

This project aims to guide developers to train a deep learning-based deepfake detection model from scratch using Python, Keras and TensorFlow. The proposed deepfake detector is based on the state-of-the-art EfficientNet structure with some customizations on the network layers, and the sample models provided were trained against a massive and comprehensive set of deepfake datasets.

# Deepfake Datasets
Due to the nature of deep neural networks being data-driven, it is necessary to acquire massive deepfake datasets with various different synthesis methods in order to achieve promising results. The following deepfake datasets were used in the final model at DF-Detect:
    DeepFake-TIMIT
    FaceForensics++
    Google Deep Fake Detection (DFD)

# Techs tag used:
    Python 3
    Keras
    TersorFlow
    EfficientNet for TensorFlow Keras
    OpenCV on Wheels
    MTCNN

# Steps Overview:
    Extract Frames from Video
    Use MTCNN for Face Detection (to crop faces from frames)
    Label Data as Real/Fake (depending on whether the frame is from a real video or a deepfake video)
    Split the Data into Training and Testing Sets
    Train a Deep Learning Model (e.g., CNN, or any other model for detecting deepfakes)