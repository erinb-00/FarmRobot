import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import message_filters
# import numpy as np
# import tensorflow as tf

# Parameters
IMG_SIZE = (64, 64)  # Resize images as needed
BATCH_SIZE = 32

# # Paths to datasets you downloaded locally after Kaggle download
# CIFAR_REFINED_PATH = './cifar'  # Replace with actual path
# RODENTS_PATH = './rodents'  # Replace with actual path

# import tensorflow as tf
# import numpy as np
# import os
# from PIL import Image
from sklearn.model_selection import train_test_split

# Parameters
IMG_SIZE = 64  # You can resize both datasets to same size
BATCH_SIZE = 32

# Load CIFAR-100 from TensorFlow datasets (as example)
# If you have files locally, you can replace this code with your own loading method
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# For CIFAR, select only classes relevant (e.g., possums, raccoons), or keep all for training
# Example: map possum (label 82), raccoon (label 70) to predator class=1, others=0
predator_classes = [70, 82]  # Example label ids for raccoon and possum in cifar-100
y_train_cifar_binary = np.isin(y_train_cifar.flatten(), predator_classes).astype(np.int32)
y_test_cifar_binary = np.isin(y_test_cifar.flatten(), predator_classes).astype(np.int32)

# Resize CIFAR images from 32x32 to 64x64 for consistency
x_train_cifar_resized = tf.image.resize(x_train_cifar, [IMG_SIZE, IMG_SIZE]).numpy()
x_test_cifar_resized = tf.image.resize(x_test_cifar, [IMG_SIZE, IMG_SIZE]).numpy()

# Load rodents images from folder - replace the path with your rodents dataset folder path
rodents_folder = '/content/rodents' #change file path accordingly
classes_rodents = ['rat', 'mouse', 'shrew']  # Example rodent classes you have
rodent_images = []
rodent_labels = []

# def load_rodent_images(folder, classes, img_size):
#     images = []
#     labels = []
#     for idx, cls in enumerate(classes):
#         cls_folder = os.path.join(folder, cls)
#         if not os.path.exists(cls_folder):
#             continue
#         for file in os.listdir(cls_folder):
#             path = os.path.join(cls_folder, file)
#             try:
#                 img = Image.open(path).convert('RGB').resize((img_size, img_size))
#                 images.append(np.array(img))
#                 labels.append(0)  # rodents = non-predators class 0
#             except:
#                 continue
#     return np.array(images), np.array(labels)

# Updated class names to match folder names
classes_rodents = ['Mice', 'Rats', 'Shrew']

def load_rodent_images(folder, classes, img_size):
    images = []
    labels = []
    for cls in classes:
        cls_folder = os.path.join(folder, cls)
        print(f"Looking in folder: {cls_folder}")
        if not os.path.exists(cls_folder):
            print(f"Folder does not exist: {cls_folder}")
            continue
        for file in os.listdir(cls_folder):
            path = os.path.join(cls_folder, file)
            try:
                img = Image.open(path).convert('RGB').resize((img_size, img_size))
                images.append(np.array(img))
                labels.append(1)  # All rodents = non-predators = class 0
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    if not images:
        print("No images loaded. Check your paths and image files.")
        return np.empty((0, img_size, img_size, 3)), np.empty((0,))
    return np.stack(images), np.array(labels)

x_rodents, y_rodents = load_rodent_images(rodents_folder, classes_rodents, IMG_SIZE)
print(f"Rodent image tensor shape: {x_rodents.shape}")
print(f"Rodent label array shape: {y_rodents.shape}")

x_rodents, y_rodents = load_rodent_images(rodents_folder, classes_rodents, IMG_SIZE)

# Combine CIFAR train and rodent images
x_train_combined = np.concatenate([x_train_cifar_resized, x_rodents], axis=0)
y_train_combined = np.concatenate([y_train_cifar_binary, y_rodents], axis=0)

# Normalize pixel values
x_train_combined = x_train_combined.astype('float32') / 255.0
x_test_cifar_resized = x_test_cifar_resized.astype('float32') / 255.0

# Split combined set into training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train_combined, y_train_combined, test_size=0.2, random_state=42)

# # Helper function to load images and labels from a directory structured per class
# def load_images_from_folder(folder_path, label_dict):
#     images = []
#     labels = []
#     for label_name, label_id in label_dict.items():
#         class_folder = os.path.join(folder_path, label_name)
#         if not os.path.exists(class_folder):
#             continue
#         for filename in os.listdir(class_folder):
#             file_path = os.path.join(class_folder, filename)
#             try:
#                 img = Image.open(file_path).convert('RGB')
#                 img = img.resize(IMG_SIZE)
#                 images.append(np.array(img))
#                 labels.append(label_id)
#             except:
#                 pass
#     return np.array(images), np.array(labels)

# # Define label mapping for the classes you want to classify as predators (or not)
# # For example, possum and raccoon as class 1 (predator), others as class 0 (non-predator/rodent)
# # Adjust labels based on dataset classes
# label_dict_cifar = {
#     'possum': 1,
#     'raccoon':1,
#     # Add other classes if desired
# }

# label_dict_rodents = {
#     'rat':1,
#     'mouse':1,
#     'shrew': 1
#     # Add other rodents if desired
# }

# # Load CIFAR refined images and labels
# # x_cifar, y_cifar = load_images_from_folder(CIFAR_REFINED_PATH, label_dict_cifar)

# # Load rodents images and labels
# x_rodent, y_rodent = load_images_from_folder(RODENTS_PATH, label_dict_rodents)

# # Combine datasets
# x_data = np.concatenate([x_cifar, x_rodent], axis=0)
# y_data = np.concatenate([y_cifar, y_rodent], axis=0)

# # Shuffle the combined dataset
# indices = np.arange(len(x_data))
# np.random.shuffle(indices)
# x_data = x_data[indices]
# y_data = y_data[indices]

# # Normalize pixel values
# x_data = x_data.astype('float32') / 255.0

# # Split dataset
# x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


# Data augmentation and generators
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

train_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
val_generator = ImageDataGenerator().flow(x_val, y_val, batch_size=BATCH_SIZE)

# Define or import your CNN or ViT model here
# For example, a simple CNN:
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the model for later inference
model.save('predator_classifier_model.h5')