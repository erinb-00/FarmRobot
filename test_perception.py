# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2

# class TestImagePublisher(Node):
#     def __init__(self, image_path):
#         super().__init__('test_image_publisher')
#         self.publisher = self.create_publisher(Image, '/camera/color/image_raw', 10)
#         self.timer = self.create_timer(1.0, self.timer_callback)
#         self.bridge = CvBridge()
#         self.image = cv2.imread(image_path)
#         self.get_logger().info(f"Loaded image from {image_path}")

#     def timer_callback(self):
#         if self.image is None:
#             self.get_logger().warn("Image is None. Check image path and mounting.")
#             return
#         msg = self.bridge.cv2_to_imgmsg(self.image, encoding='bgr8')
#         self.publisher.publish(msg)
#         self.get_logger().info("Published test image.")

# def main(args=None):
#     rclpy.init(args=args)
#     image_path = './kangaroo-facts.jpg'
#     node = TestImagePublisher(image_path)
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TestRGBDPublisher(Node):
    def __init__(self, rgb_path):
        super().__init__('test_rgbd_publisher')
        self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)

        self.bridge = CvBridge()
        self.rgb_image = cv2.imread(rgb_path)

        # Create synthetic depth image (same resolution, float32)
        height, width = self.rgb_image.shape[:2]
        self.depth_image = np.full((height, width), 1.5, dtype=np.float32)  # constant depth of 1.5 meters

        self.timer = self.create_timer(1.0, self.publish_images)

        self.get_logger().info("Test RGB-D publisher started")

    def publish_images(self):
        now = self.get_clock().now().to_msg()

        # Publish RGB
        rgb_msg = self.bridge.cv2_to_imgmsg(self.rgb_image, encoding='bgr8')
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = "camera_color_optical_frame"
        self.rgb_pub.publish(rgb_msg)

        # Publish Depth
        depth_msg = self.bridge.cv2_to_imgmsg(self.depth_image, encoding='32FC1')
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = "camera_depth_optical_frame"
        self.depth_pub.publish(depth_msg)

        self.get_logger().info("Published RGB and depth frames.")

def main(args=None):
    rclpy.init(args=args)
    node = TestRGBDPublisher('./pngtree-mineral-water-bottles-png-image_12926881.png')  # Replace with your image path
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
