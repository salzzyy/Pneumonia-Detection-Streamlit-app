import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import save_model

# Set image dimensions and batch size
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 12

# Set paths for the training and validation data
TRAIN_DIR = "D:/Pneumonia-Detection/chest_xray/train"
VAL_DIR = "D:/Pneumonia-Detection/chest_xray/val"
TEST_DIR = "D:/Pneumonia-Detection/chest_xray/test"


# train_dir = r"D:\pneumonia_detction\chest_xray\train"
# val_dir = r"D:\pneumonia_detction\chest_xray\val"

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Prepare the data generators
train_data_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="grayscale",
)

val_data_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="grayscale",
)

# Build the CNN model
model = Sequential()

# Add layers as per your model architecture
model.add(
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 1), padding="same")
)
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer=RMSprop(), loss="binary_crossentropy", metrics=["accuracy"])

# Set up learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, verbose=1, factor=0.3, min_lr=0.000001
)

# Train the model
history = model.fit(
    train_data_gen,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    callbacks=[learning_rate_reduction],
)

# Save the trained model
model.save("pneumonia_model.h5")

# Plot training history (accuracy & loss)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Plot")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Save the accuracy/loss plot
plt.savefig("training_history.png")
plt.close()


# test_loss, test_acc = model.evaluate(test_data_gen, verbose=2)
# print(f"Test Accuracy: {test_acc*100}%")

model.save("pneumonia_detection_model.h5")

# Load the saved model
# Save the Model
os.makedirs("models", exist_ok=True)
model.save("models/pneumonia_model.h5")
print("Model saved to 'models/pneumonia_model.h5'")
