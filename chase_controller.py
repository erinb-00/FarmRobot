#!/usr/bin/env python
"""
Chase Controller Node for ROS2

This node implements a finite state machine to control a robot's behavior
when chasing and deterring targets. It uses PID controllers for motion control
and maintains a 1m distance from detected targets.

Author: Gary Ding
Date: 5/27/25
"""

import math
import time
from enum import Enum
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool

# Constants
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
DEFAULT_SCAN_TOPIC = 'scan'
DEFAULT_TARGET_POSE_TOPIC = 'target_pose'
DEFAULT_SERVICE_NAME = 'chase_on_off'

FREQUENCY = 10  # Hz
TARGET_DISTANCE = 1.0  # m - desired distance from target
DISTANCE_TOLERANCE = 0.1  # m - tolerance for target distance
VELOCITY_TOLERANCE = 0.05  # m/s - threshold for considering target stationary

# PID gains for linear motion (distance control)
LINEAR_KP = 1.0
LINEAR_KI = 0.1
LINEAR_KD = 0.05

# PID gains for angular motion (heading control)
ANGULAR_KP = 2.0
ANGULAR_KI = 0.1
ANGULAR_KD = 0.1

# Motion limits
MAX_LINEAR_VELOCITY = 0.5  # m/s
MAX_ANGULAR_VELOCITY = 1.0  # rad/s

# Deterrent behavior parameters
DETERRENT_CIRCLES = 2
DETERRENT_RADIUS = 1.5  # m
DETERRENT_ANGULAR_VELOCITY = 0.5  # rad/s

# Timeout parameters
TARGET_TIMEOUT = 5.0  # seconds - time without target before returning to patrol
STATIONARY_TIMEOUT = 3.0  # seconds - time target must be stationary before deterrent

USE_SIM_TIME = True


class ChaseState(Enum):
    """Finite State Machine states for chase controller."""
    STOP = 0
    PATROL = 1
    CHASE = 2
    DETERRENT = 3


class PIDController:
    """Simple PID controller implementation."""
    
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None
    
    def update(self, error: float, current_time: float) -> float:
        """Update PID controller and return control output."""
        if self.last_time is None:
            self.last_time = current_time
            return 0.0
        
        dt = current_time - self.last_time
        if dt <= 0.0:
            return 0.0
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.previous_error) / dt
        
        # Update for next iteration
        self.previous_error = error
        self.last_time = current_time
        
        return proportional + integral + derivative
    
    def reset(self):
        """Reset PID controller state."""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None


class ChaseController(Node):
    """ROS2 node implementing chase controller with finite state machine."""
    
    def __init__(self, node_name: str = "chase_controller"):
        super().__init__(node_name)
        
        # Set up simulation time parameter
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])
        
        # Publishers and subscribers
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        self._laser_sub = self.create_subscription(
            LaserScan, DEFAULT_SCAN_TOPIC, self._laser_callback, 1)
        self._target_sub = self.create_subscription(
            Pose, DEFAULT_TARGET_POSE_TOPIC, self._target_callback, 1)
        
        # Service for enabling/disabling chase behavior
        self._service = self.create_service(
            SetBool, f'{node_name}/{DEFAULT_SERVICE_NAME}', self._service_callback)
        
        # State machine
        self._state = ChaseState.STOP
        self._previous_state = ChaseState.STOP
        
        # Target tracking
        self._target_pose: Optional[Pose] = None
        self._last_target_time: Optional[float] = None
        self._target_stationary_start: Optional[float] = None
        self._previous_target_position: Optional[Point] = None
        
        # PID controllers
        self._linear_pid = PIDController(LINEAR_KP, LINEAR_KI, LINEAR_KD)
        self._angular_pid = PIDController(ANGULAR_KP, ANGULAR_KI, ANGULAR_KD)
        
        # Deterrent behavior state
        self._deterrent_start_time: Optional[float] = None
        self._deterrent_circles_completed = 0
        self._deterrent_start_angle: Optional[float] = None
        
        # Obstacle detection
        self._obstacle_detected = False
        self._min_obstacle_distance = 0.3  # m
        
        # Timer for main control loop
        self._timer = self.create_timer(1.0 / FREQUENCY, self._control_loop)
        
        self.get_logger().info("Chase Controller initialized")
    
    def _service_callback(self, request, response):
        """Handle service calls to enable/disable chase behavior."""
        if request.data:
            if self._state == ChaseState.STOP:
                self._state = ChaseState.PATROL
                self._reset_controllers()
                response.success = True
                response.message = "Chase controller activated"
                self.get_logger().info("Chase controller activated")
            else:
                response.success = False
                response.message = "Chase controller already active"
        else:
            self._state = ChaseState.STOP
            self._stop_robot()
            response.success = True
            response.message = "Chase controller stopped"
            self.get_logger().info("Chase controller stopped")
        
        return response
    
    def _target_callback(self, msg: Pose):
        """Handle target pose messages from perception node."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Check if target is moving
        if self._previous_target_position is not None:
            distance_moved = self._calculate_distance(
                self._previous_target_position, msg.position)
            
            if distance_moved < VELOCITY_TOLERANCE:
                # Target appears stationary
                if self._target_stationary_start is None:
                    self._target_stationary_start = current_time
            else:
                # Target is moving
                self._target_stationary_start = None
        
        self._target_pose = msg
        self._last_target_time = current_time
        self._previous_target_position = msg.position
        
        # Transition to chase state if in patrol
        if self._state == ChaseState.PATROL:
            self._state = ChaseState.CHASE
            self._reset_controllers()
            self.get_logger().info("Target detected, entering CHASE state")
    
    def _laser_callback(self, msg: LaserScan):
        """Handle laser scan messages for obstacle detection."""
        # Check for obstacles in front of robot
        front_angles = []
        for i, range_val in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            # Check angles roughly in front of robot (±30 degrees)
            if -math.pi/6 <= angle <= math.pi/6:
                if msg.range_min <= range_val <= msg.range_max:
                    front_angles.append(range_val)
        
        if front_angles:
            min_distance = min(front_angles)
            self._obstacle_detected = min_distance < self._min_obstacle_distance
        else:
            self._obstacle_detected = False
    
    def _control_loop(self):
        """Main control loop implementing the finite state machine."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # State transitions and actions
        if self._state == ChaseState.STOP:
            self._stop_robot()
        
        elif self._state == ChaseState.PATROL:
            self._patrol_behavior()
            self._check_target_timeout(current_time)
        
        elif self._state == ChaseState.CHASE:
            if self._target_pose is None:
                self._state = ChaseState.PATROL
                self.get_logger().info("No target, returning to PATROL")
            elif self._check_target_timeout(current_time):
                self._state = ChaseState.PATROL
                self.get_logger().info("Target timeout, returning to PATROL")
            elif self._should_enter_deterrent(current_time):
                self._state = ChaseState.DETERRENT
                self._deterrent_start_time = current_time
                self._deterrent_circles_completed = 0
                self._deterrent_start_angle = None
                self.get_logger().info("Target stationary, entering DETERRENT state")
            else:
                self._chase_behavior(current_time)
        
        elif self._state == ChaseState.DETERRENT:
            if self._deterrent_behavior(current_time):
                # Deterrent complete, return to patrol
                self._state = ChaseState.PATROL
                self._target_pose = None
                self._reset_controllers()
                self.get_logger().info("Deterrent complete, returning to PATROL")
    
    def _patrol_behavior(self):
        """Implement patrol behavior (simple forward motion with obstacle avoidance)."""
        if self._obstacle_detected:
            # Turn to avoid obstacle
            self._publish_velocity(0.0, 0.5)
        else:
            # Move forward slowly
            self._publish_velocity(0.2, 0.0)
    
    def _chase_behavior(self, current_time: float):
        """Implement chase behavior using PID control to maintain target distance."""
        if self._target_pose is None:
            return
        
        # Calculate distance and angle to target
        target_distance = math.sqrt(
            self._target_pose.position.x**2 + self._target_pose.position.y**2)
        target_angle = math.atan2(
            self._target_pose.position.y, self._target_pose.position.x)
        
        # Distance error (positive means too far, negative means too close)
        distance_error = target_distance - TARGET_DISTANCE
        
        # Angular error (angle to target)
        angular_error = target_angle
        
        # PID control
        linear_output = self._linear_pid.update(distance_error, current_time)
        angular_output = self._angular_pid.update(angular_error, current_time)
        
        # Apply limits and obstacle avoidance
        if self._obstacle_detected and distance_error > 0:
            # Stop if obstacle detected while approaching
            linear_velocity = 0.0
            angular_velocity = 0.5  # Turn to avoid obstacle
        else:
            linear_velocity = max(-MAX_LINEAR_VELOCITY, 
                                min(MAX_LINEAR_VELOCITY, linear_output))
            angular_velocity = max(-MAX_ANGULAR_VELOCITY, 
                                 min(MAX_ANGULAR_VELOCITY, angular_output))
        
        self._publish_velocity(linear_velocity, angular_velocity)
    
    def _deterrent_behavior(self, current_time: float) -> bool:
        """
        Implement deterrent behavior (circle around target).
        Returns True when deterrent is complete.
        """
        if self._target_pose is None:
            return True
        
        # Calculate position relative to target for circling
        target_x = self._target_pose.position.x
        target_y = self._target_pose.position.y
        
        # Calculate current angle around target
        current_angle = math.atan2(-target_y, -target_x)  # Angle from target to robot
        
        if self._deterrent_start_angle is None:
            self._deterrent_start_angle = current_angle
        
        # Calculate how much we've rotated
        angle_diff = current_angle - self._deterrent_start_angle
        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Check if we've completed a full circle
        if abs(angle_diff) >= 2 * math.pi:
            self._deterrent_circles_completed += 1
            self._deterrent_start_angle = current_angle
            
            if self._deterrent_circles_completed >= DETERRENT_CIRCLES:
                return True
        
        # Move in a circle around the target
        # Maintain deterrent radius while circling
        target_distance = math.sqrt(target_x**2 + target_y**2)
        distance_error = target_distance - DETERRENT_RADIUS
        
        # Simple control to maintain circular motion
        linear_velocity = 0.1 * distance_error  # Adjust distance to target
        angular_velocity = DETERRENT_ANGULAR_VELOCITY  # Constant angular velocity for circling
        
        # Apply limits
        linear_velocity = max(-MAX_LINEAR_VELOCITY, 
                            min(MAX_LINEAR_VELOCITY, linear_velocity))
        
        self._publish_velocity(linear_velocity, angular_velocity)
        return False
    
    def _should_enter_deterrent(self, current_time: float) -> bool:
        """Check if robot should enter deterrent state."""
        if self._target_stationary_start is None:
            return False
        
        return (current_time - self._target_stationary_start) >= STATIONARY_TIMEOUT
    
    def _check_target_timeout(self, current_time: float) -> bool:
        """Check if target has timed out."""
        if self._last_target_time is None:
            return False
        
        return (current_time - self._last_target_time) > TARGET_TIMEOUT
    
    def _publish_velocity(self, linear: float, angular: float):
        """Publish velocity command."""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self._cmd_pub.publish(twist)
    
    def _stop_robot(self):
        """Stop the robot."""
        self._publish_velocity(0.0, 0.0)
    
    def _reset_controllers(self):
        """Reset PID controllers."""
        self._linear_pid.reset()
        self._angular_pid.reset()
    
    def _calculate_distance(self, point1: Point, point2: Point) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2 + 
            (point1.z - point2.z)**2
        )


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    chase_controller = ChaseController()
    
    try:
        rclpy.spin(chase_controller)
    except KeyboardInterrupt:
        chase_controller.get_logger().info("Chase controller interrupted")
    finally:
        chase_controller._stop_robot()
        chase_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 