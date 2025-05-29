import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
# import numpy as np
# import tensorflow as tf

# Parameters
IMG_SIZE = (64, 64)  # Resize images as needed
BATCH_SIZE = 32

# Paths to datasets you downloaded locally after Kaggle download
CIFAR_REFINED_PATH = '/Users/tofaratifolayan/Desktop/vnc-ros/workspace/src/cifar'  # Replace with actual path
RODENTS_PATH = '/Users/tofaratifolayan/Desktop/vnc-ros/workspace/src/rodents'  # Replace with actual path

# Helper function to load images and labels from a directory structured per class
def load_images_from_folder(folder_path, label_dict):
    images = []
    labels = []
    for label_name, label_id in label_dict.items():
        class_folder = os.path.join(folder_path, label_name)
        if not os.path.exists(class_folder):
            continue
        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(label_id)
            except:
                pass
    return np.array(images), np.array(labels)

# Define label mapping for the classes you want to classify as predators (or not)
# For example, possum and raccoon as class 1 (predator), others as class 0 (non-predator/rodent)
# Adjust labels based on dataset classes
label_dict_cifar = {
    'possum': 1,
    'raccoon':1,
    # Add other classes if desired
}

label_dict_rodents = {
    'rat':0,
    'mouse':0,
    # Add other rodents if desired
}

# Load CIFAR refined images and labels
x_cifar, y_cifar = load_images_from_folder(CIFAR_REFINED_PATH, label_dict_cifar)

# Load rodents images and labels
x_rodent, y_rodent = load_images_from_folder(RODENTS_PATH, label_dict_rodents)

# Combine datasets
x_data = np.concatenate([x_cifar, x_rodent], axis=0)
y_data = np.concatenate([y_cifar, y_rodent], axis=0)

# Shuffle the combined dataset
indices = np.arange(len(x_data))
np.random.shuffle(indices)
x_data = x_data[indices]
y_data = y_data[indices]

# Normalize pixel values
x_data = x_data.astype('float32') / 255.0

# Split dataset
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

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
    tf.keras.layers.InputLayer(input_shape=IMG_SIZE+(3,)),
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

class PredatorClassifierNode(Node):
    def __init__(self):
        super().__init__('predator_classifier_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # Adjust topic as per your camera
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Load model
        self.model = tf.keras.models.load_model('predator_classifier_model.h5')

        # Expected input size
        self.img_size = (64, 64)  # same as training

        self.get_logger().info("Predator classifier node started.")

    def preprocess(self, cv_image):
        # Resize, normalize etc.
        img = cv2.resize(cv_image, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)  # batch size 1
        return img

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image (BGR)
            cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert BGR to RGB as model was trained on RGB images
            cv_image_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        # Preprocess image
        input_img = self.preprocess(cv_image_rgb)

        # Predict
        pred = self.model.predict(input_img)
        prob = pred[0][0]  # sigmoid output probability

        # Assuming threshold 0.5
        is_predator = prob > 0.5

        # Log or publish result
        self.get_logger().info(f"Predator probability: {prob:.3f} => {'Predator' if is_predator else 'Not Predator'}")

def main(args=None):
    rclpy.init(args=args)
    node = PredatorClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()