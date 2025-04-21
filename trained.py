import os
import json
import shutil
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import Xception  # Using XceptionNet for deepfake detection
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ✅ Check TensorFlow version
print('TensorFlow version:', tf.__version__)

# ✅ Set paths (Modify as needed)
drive_path = "/content/drive/My Drive/myDataset/"
dataset_path = os.path.join(drive_path, "split_dataset")
checkpoint_path = os.path.join(drive_path, "checkpoints")
os.makedirs(checkpoint_path, exist_ok=True)

# ✅ Define paths for train, validation, and test datasets
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

# ✅ Image parameters
input_size = (128, 128)  # Resize all images to 128x128
batch_size = 32

# ✅ Function to load dataset while keeping folder structure
def load_image_dataset(directory, batch_size=32, img_size=(128, 128), shuffle=True):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size=batch_size,
        image_size=img_size,
        label_mode="binary",  # Ensures binary classification
        shuffle=shuffle
    )
    return dataset

# ✅ Load datasets
train_dataset = load_image_dataset(train_path, batch_size, input_size)
val_dataset = load_image_dataset(val_path, batch_size, input_size)
test_dataset = load_image_dataset(test_path, batch_size, input_size, shuffle=False)

# ✅ Prefetch for better performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# ✅ Build Xception model
base_model = Xception(
    weights="imagenet",
    input_shape=(input_size[0], input_size[1], 3),
    include_top=False
)

# ✅ Freeze base model (optional)
base_model.trainable = False

# ✅ Define the classifier
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification (Real vs Fake)
])

# ✅ Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ✅ Callbacks for early stopping and model saving
checkpoint_filepath = os.path.join(checkpoint_path, "best_model.h5")

callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1),
    ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_loss", mode="min", verbose=1, save_best_only=True)
]

# ✅ Train the model
num_epochs = 20
history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=callbacks
)

# ✅ Save the final model in Google Drive
final_model_path = os.path.join(checkpoint_path, "final_model.h5")
model.save(final_model_path)
print(f"Final model saved at: {final_model_path}")

# ✅ Load the best saved model for evaluation
best_model = load_model(checkpoint_filepath)

# ✅ Generate predictions on test set
preds = best_model.predict(test_dataset, verbose=1)

# ✅ Save test predictions
test_results = pd.DataFrame({
    "Filename": test_dataset.file_paths,
    "Prediction": preds.flatten()
})
test_results.to_csv(os.path.join(drive_path, "test_predictions.csv"), index=False)
print("Predictions saved!")

# ✅ Optional: Plot training history
import matplotlib.pyplot as plt

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, "bo", label="Training Accuracy")
plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.show()