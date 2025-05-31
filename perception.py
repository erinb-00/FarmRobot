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

class PredatorClassifierNode(Node):
    def __init__(self):
        super().__init__('predator_classifier_node')
        self.bridge = CvBridge()
        # self.subscription = self.create_subscription(
        #     Image,
        #     '/camera/color/image_raw', 
        #     self.image_callback,
        #     10)
        # self.subscription  # prevent unused variable warning
        
        # self.depth_subscription = self.create_subscription(
        #     Image,
        #     '/camera/depth/image_raw', 
        #     self.image_callback_depth,
        #     10)

        # # Load model
        # self.model = tf.keras.models.load_model('predator_classifier_model_rodents_pred.h5')

        # # Expected input size
        # self.img_size = (64, 64)  # same as training

        # self.get_logger().info("Predator classifier node started.")
         # Load model
        self.model = tf.keras.models.load_model('predator_classifier_model_rodents_pred.h5')
        self.get_logger().info("Predator classifier node started.")

        # Sync RGB and depth messages
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')

        # Time synchronizer
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.synced_callback)

    def preprocess(self, cv_image):
        # Resize, normalize etc.
        img = cv2.resize(cv_image, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)  # batch size 1
        return img
    
    def synced_callback(self, rgb_msg, depth_msg):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2RGB)

            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')  # usually 32FC1
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        # Run classifier
        input_img = self.preprocess(cv_rgb)
        pred = self.model.predict(input_img)
        prob = pred[0][0]
        is_predator = prob > 0.5

        if is_predator:
            cx, cy = self.estimate_intruder_position(cv_rgb)
            if cx == -1:
                self.get_logger().warn("Could not locate intruder in image.")
                return

            # Get depth at (cx, cy)
            depth = cv_depth[cy, cx]
            if np.isnan(depth) or depth == 0:
                self.get_logger().warn(f"Invalid depth at ({cx},{cy})")
                return

            # Estimate 3D position from depth
            x, y, z = self.project_to_3d(cx, cy, depth)
            self.get_logger().info(f"Predator at image ({cx}, {cy}), 3D position (x={x:.2f}, y={y:.2f}, z={z:.2f}) prob={prob:.2f}")
            return x, y, z
        else:
            self.get_logger().info(f"Not Predator (prob={prob:.2f})")

    def project_to_3d(self, u, v, depth):
        # Camera intrinsics (replace with actual values from /camera/color/camera_info)
        fx = 600  # focal length in pixels
        fy = 600
        cx = 320  # principal point
        cy = 240

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return x, y, z

    # def image_callback(self, msg):
    #     try:
    #         # Convert ROS Image message to OpenCV image (BGR)
    #         cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         # Convert BGR to RGB as model was trained on RGB images
    #         cv_image_rgb = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
    #     except Exception as e:
    #         self.get_logger().error(f"CV bridge error: {e}")
    #         return

    #     # Preprocess image
    #     input_img = self.preprocess(cv_image_rgb)

    #     # Predict
    #     pred = self.model.predict(input_img)
    #     prob = pred[0][0]  # sigmoid output probability

    #     # Assuming threshold 0.5
    #     is_predator = prob > 0.5

    #     # Log or publish result
    #     self.get_logger().info(f"Predator probability: {prob:.3f} => {'Predator' if is_predator else 'Not Predator'}")
        
        # # Display text on image (using BGR version for display)
        # display_text = f"Predator Prob: {prob:.2f} - {'YES' if is_predator else 'NO'}"
        # cv2.putText(cv_image_bgr, display_text, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_predator else (0, 255, 0), 2)

        # # Show the image in a window
        # cv2.imshow("ROSbot Camera - Predator Detection", cv_image_bgr)
        # cv2.waitKey(1)  # 1ms delay to process UI events

        # self.get_logger().info(display_text)

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