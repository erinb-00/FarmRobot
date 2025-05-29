import rclpy # module for ROS APIs
from rclpy.node import Node
#import rospy
from rclpy.duration import Duration

import tf2_ros # library for transformations.
from tf2_ros import TransformException
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped,Twist,PoseArray,PointStamped
from tf_transformations import euler_from_quaternion
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header,String
import numpy as np
#from skimage.draw import bresenham
#import pybresenham
#from bresenham import bresenham
import math

# Frequency at which the loop operates
FREQUENCY = 10 #Hz.
VELOCITY = 0.2 #m/s

# Velocities that will be used (TODO: feel free to tune)
NODE_NAME = "test_tf"
LINEAR_VELOCITY = 0.2 # m/s
ANGULAR_VELOCITY = math.pi/4 # rad/s
DEFAULT_SCAN_TOPIC = 'scan'
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
TF_BASE_LINK = 'base_link'
TF_LASER_LINK = 'laser'
detection_topic = '/detected_status'
roboloc = '/trackrobot'
visual = '/seerobot'

USE_SIM_TIME = True

class Intruder(Node):
    def __init__(self):
        """Constructor."""
        super().__init__(node_name=NODE_NAME)

        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])
        self.linear_velocity = VELOCITY
        self.angular_velocity = ANGULAR_VELOCITY
        self.robot_trans = None
        self.visionfy = False
        self.detected = False

        #publishing messages
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        self.detpub = self.create_publisher(String,detection_topic,1)

        # Setting up transformation listener.
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        #listening for messages
        #scan
        self.create_subscription(
            LaserScan,
            DEFAULT_SCAN_TOPIC,
            self._laser_callback,
            10
        )

        #robot location message
        self.create_subscription(
            TransformStamped,
            roboloc,
            self.robloc_callback,
            10
        )

        #robot vision confirmation
        self.create_subscription(
            String,
            visual,
            self.visionT,
            1
        )

        self.last_time = self.get_clock().now()

    def _laser_callback(self, msg):
        self.get_logger().info("Laser callback triggered.")
        current_time = self.get_clock().now()

        laser_ranges = msg.ranges
        stamp = msg.header.stamp

    def robloc_callback(self,msg:TransformStamped):
        self.robot_trans = msg.transform
        self.get_logger().info(f"Received transform: {self.robot_trans}")

    def visionT(self,msg:String):
        self.visionfy = msg.string
        self.get_logger().info(f"Vision confirmation received: {self.visionfy}")

    def stop(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self._cmd_pub.publish(stop_msg)

    def move_forward(self, duration):
        """Function to move_forward for a given duration."""
        # Setting velocities. 
        twist_msg = Twist()
        twist_msg.linear.x = self.linear_velocity

        duration = Duration(seconds=duration)
        rclpy.spin_once(self)
        start_time = self.get_clock().now()

        # Loop.
        while rclpy.ok():
            rclpy.spin_once(self)
            # Check if traveled of given distance based on time.
            self.get_logger().info(f"{start_time} {self.get_clock().now()} {duration}")
            if self.get_clock().now() - start_time >= duration:
                break

            # Publish message.
            self._cmd_pub.publish(twist_msg)

        # Traveled the required distance, stop.
        self.stop()
    
    def spin(self, angle):
        """Rotates the robot by the given angle."""
        twist_msg = Twist()
        #
        # twist_msg.angular.z = 0.5 * angle 
        start_time = self.get_clock().now().nanoseconds
        # duration = abs(angle / 0.5) * 1e9 
        #alt
        angular_speed = 0.5  # rad/s
        twist_msg.angular.z = angular_speed if angle > 0 else -angular_speed
        duration = abs(angle) / angular_speed * 1e9  

        #
        while rclpy.ok() and self.get_clock().now().nanoseconds - start_time < duration:
            self._cmd_pub.publish(twist_msg)
            rclpy.spin_once(self)
        twist_msg.angular.z = 0.0
        self._cmd_pub.publish(twist_msg)
        rclpy.spin_once(self)

        #function that manages movement of robot from vertex to vertex
    def driver(self,vertices,stamp):
        #vertices.append(vertices[0])
        for x,y in vertices:
            print("driving to destination")
            x = float(x)
            y = float(y)
            self.porter(x,y,stamp)
            self.grid_publisher() #publishing map to rviz

      #function that drives robot to a destination given a specific coordinate
    def porter(self,xpt,ypt,stamp):
        try:
            # Getting transform from odom to base_link
            #now = rclpy.time.Time()
            now = stamp
            #trans = self.tf_buffer.lookup_transform('base_link', 'odom', now)
            while not self.tf_buffer.can_transform('odom', 'base_link', now):
                rclpy.spin_once(self,timeout_sec=0.1)
                #pass
            trans = self.tf_buffer.lookup_transform('odom','base_link', now)

            # Transforming the global point to robot's local frame
            target_point = PointStamped()
            target_point.header.frame_id = 'odom'
            target_point.header.stamp = now.to_msg()
            target_point.point.x = xpt
            target_point.point.y = ypt
            target_point.point.z = 0.0

            #local_target = self.tf_buffer.transform(target_point, 'base_link')
            #transform = self.tf_buffer.lookup_transform('base_link', 'odom', rclpy.time.Time())
            transform = self.tf_buffer.lookup_transform('base_link', 'odom', now)
            local_target = do_transform_point(target_point, transform)

            # Distance and angle to the target in robot's frame
            dx = local_target.point.x
            dy = local_target.point.y
            distance = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)

            twist = Twist()

            if distance > 0.05:
                # Rotating toward the point
                twist.angular.z = self.angular_velocity * angle
                twist.linear.x = self.linear_velocity * math.cos(angle)
                self._cmd_pub.publish(twist)
                #alt
                self.spin(angle)
                duration = distance/self.linear_velocity
                self.move_forward(duration)
            else:
                self.stop()
        except Exception as e:
            self.get_logger().warn(f"TF error: {e}")
            rclpy.spin_once(self, timeout_sec=0.1)

    def getpose(self, target_frame, source_frame,stamp):
        try:
            
            now = stamp
            #trans = self.tf_buffer.lookup_transform('base_link', 'odom', now)
            while not self.tf_buffer.can_transform(target_frame, source_frame, now):
                rclpy.spin_once(self,timeout_sec=0.1)
                #pass

            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, now)

            #
            world_x = transform.transform.translation.x
            world_y = transform.transform.translation.y
            yaw = self.get_yaw_from_quaternion(transform.transform.rotation)


            return world_x, world_y,yaw

        except Exception as e:
            self.get_logger().warn(f"Could not get x, y from {target_frame} to {source_frame}: {e}")
            return None,None,None

    
    def interact(self,stamp,thresh):
        robotloc = self.robot_trans 
        robox = robotloc.transform.translation.x
        roboy = robotloc.transform.translation.y
        robo_yaw = self.get_yaw_from_quaternion(robotloc.transform.rotation)

        mylocx,mylocy,_  = self.getpose("odom", "base_link",stamp)

        #computing distance between robots
        dist = math.sqrt((robox - mylocx)**2 + (roboy - mylocy)**2)
        if dist < thresh and self.visionfy == True: #detected message is published if robot can see intruder and it is in acceptable proximity
            det_msg = String()
            det_msg.data = 'Detected'
            self.detpub.publish(det_msg)
            self.detected = True
        
    #need map to get location to move intruder to 

#Running functions
def main(args=None):
    rclpy.init(args=args)
    stamp = rclpy.time.Time()
    intrude = Intruder()
    
    #Moving intruder to location
    print(stamp)
    vertices = None
    intrude.driver(vertices,stamp)
    
    #interacting with nearby robot
    thresh = 1
    robotloc = None
    intrude.interact(stamp,thresh)


    intrude.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()