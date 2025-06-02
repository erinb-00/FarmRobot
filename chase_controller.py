#!/usr/bin/env python
"""
Chase Controller Node for ROS2

This node implements chase controlling mechanism for autonomous robot behavior.
It uses PID controllers for motion control, occupancy grid mapping for obstacle 
awareness, and BFS path planning to navigate around obstacles while maintaining 
a 1m distance from detected targets. The finite state machine is handled by 
an external node.

Author: Gary Ding
Date: 6/1/25
"""

import math
import time
import os
import threading
from typing import Optional, Tuple, List
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, Pose, Point, TransformStamped
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool
from std_msgs.msg import String, Bool, Empty
from tf2_ros import TransformBroadcaster, TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf_transformations

# Try to import pygame for audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Sound will be simulated only.")

# Constants
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
DEFAULT_SCAN_TOPIC = 'base_scan'
DEFAULT_TARGET_POSE_TOPIC = 'target_pose'
DEFAULT_OCCUPANCY_GRID_TOPIC = 'map'
DEFAULT_CHASE_ENABLE_TOPIC = 'chase_enable'
DEFAULT_CHASE_MODE_TOPIC = 'chase_mode'
DEFAULT_CHASE_STATUS_TOPIC = 'chase_status'
DEFAULT_SOUND_TOPIC = 'play_sound'
DEFAULT_BASE_FRAME = 'base_link'
DEFAULT_WORLD_FRAME = 'map'

# Sound file path (relative to the workspace)
ALARM_SOUND_FILE = "src/Alarm Sound Effect.mp3"

FREQUENCY = 10  # Hz
TARGET_DISTANCE = 1.0  # m - desired distance from target
DISTANCE_TOLERANCE = 0.1  # m - tolerance for target distance

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
MAX_ANGULAR_VELOCITY = 0.7  # rad/s

# Deterrent behavior parameters
DETERRENT_SOUND_DISTANCE = 2.0  # m - distance at which to start playing sound
SOUND_INTERVAL = 2.0  # seconds - minimum interval between sound plays

# Obstacle avoidance parameters
MIN_OBSTACLE_DISTANCE = 0.5  # m - minimum distance to obstacles
OBSTACLE_AVOIDANCE_ANGULAR_VEL = 0.3  # rad/s - turning speed when avoiding obstacles
FRONT_SCAN_ANGLE = math.pi / 6  # ±30 degrees for front obstacle detection

# Timeout parameters
TARGET_TIMEOUT = 5.0  # seconds - time without target before reporting lost

# Testing mode parameters
TESTING_MODE = True  # Set to True to enable integrated testing

# State transition parameters
CHASE_TO_DETERRENT_DISTANCE = 1.0  # m - distance at which to switch from chase to deterrent

USE_SIM_TIME = True


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
    """ROS2 node implementing chase controller mechanism."""
    
    def __init__(self, node_name: str = "chase_controller"):
        super().__init__(node_name)
        
        # Set up simulation time parameter
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])
        
        # Set up TF2 listener (similar to pa3)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publishers and subscribers
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        self._status_pub = self.create_publisher(String, DEFAULT_CHASE_STATUS_TOPIC, 1)
        self._sound_pub = self.create_publisher(Empty, DEFAULT_SOUND_TOPIC, 1)
        
        # State output publishers for external FSM
        self._state_pub = self.create_publisher(String, 'chase_state', 1)
        self._fsm_status_pub = self.create_publisher(String, 'fsm_status', 1)
        
        self._laser_sub = self.create_subscription(
            LaserScan, DEFAULT_SCAN_TOPIC, self._laser_callback, 1)
        self._target_sub = self.create_subscription(
            Pose, DEFAULT_TARGET_POSE_TOPIC, self._target_callback, 1)
        self._enable_sub = self.create_subscription(
            Bool, DEFAULT_CHASE_ENABLE_TOPIC, self._enable_callback, 1)
        self._mode_sub = self.create_subscription(
            String, DEFAULT_CHASE_MODE_TOPIC, self._mode_callback, 1)
        
        # Transform broadcaster for robot pose
        self._tf_broadcaster = TransformBroadcaster(self)
        
        # Chase controller state
        self._chase_enabled = False
        self._chase_mode = "chase"  # "chase" or "deterrent"
        
        # Robot pose tracking (using TF2 like pa3)
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_theta = 0.0
        self._last_cmd_time = None
        self._last_linear_vel = 0.0
        self._last_angular_vel = 0.0
        
        # Target tracking
        self._target_pose: Optional[Pose] = None
        self._last_target_time: Optional[float] = None
        
        # PID controllers
        self._linear_pid = PIDController(LINEAR_KP, LINEAR_KI, LINEAR_KD)
        self._angular_pid = PIDController(ANGULAR_KP, ANGULAR_KI, ANGULAR_KD)
        
        # Sound control for deterrent behavior
        self._last_sound_time: Optional[float] = None
        self._sound_playing = False
        
        # Initialize pygame for audio playback
        self._sound_initialized = False
        self._alarm_sound_path = None
        self._init_sound_system()
        
        # Obstacle detection
        self._obstacle_detected = False
        
        # Testing mode state
        self._testing_mode = TESTING_MODE
        self._test_running = False
        self._current_test_target = None
        
        # Time-to-Clear (TTC) tracking
        self._chase_start_time: Optional[float] = None
        self._ttc_measurements: List[float] = []
        self._current_test_number = 0
        
        # Timer for main control loop
        self._timer = self.create_timer(1.0 / FREQUENCY, self._control_loop)
        
        # Allow time for ROS initialization and TF buffer to fill (like pa3)
        self.get_logger().info("Initializing chase controller...")
        time.sleep(1.0)
        
        self.get_logger().info("Chase Controller initialized")
        
        # Start testing interface if enabled
        if self._testing_mode:
            self._start_testing_interface()
    
    def _init_sound_system(self):
        """Initialize the sound system and load alarm sound."""
        if not PYGAME_AVAILABLE:
            self.get_logger().warn("Pygame not available - will use simulated sound fallback")
            self._sound_initialized = False
            return
        
        try:
            # Initialize pygame mixer
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Find the alarm sound file
            # Try different possible paths
            possible_paths = [
                ALARM_SOUND_FILE,  # Relative to workspace
                os.path.join(os.getcwd(), ALARM_SOUND_FILE),  # Absolute from current dir
                os.path.join(os.path.dirname(__file__), "..", "Alarm Sound Effect.mp3"),  # Relative to script
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self._alarm_sound_path = path
                    break
            
            if self._alarm_sound_path:
                # Test loading the sound
                pygame.mixer.music.load(self._alarm_sound_path)
                self._sound_initialized = True
                self.get_logger().info(f"Sound system initialized with alarm file: {self._alarm_sound_path}")
            else:
                self.get_logger().warn(f"Alarm sound file not found. Will use simulated sound fallback.")
                self.get_logger().warn(f"Searched paths: {possible_paths}")
                self._sound_initialized = False
                
        except Exception as e:
            self.get_logger().error(f"Failed to initialize sound system: {e}")
            self.get_logger().warn("Will use simulated sound fallback")
            self._sound_initialized = False
    
    def _enable_callback(self, msg: Bool):
        """Handle chase enable/disable commands."""
        self._chase_enabled = msg.data
        if not self._chase_enabled:
            self._stop_robot()
            self._reset_controllers()
            self._publish_status("disabled")
            self.get_logger().info("Chase controller disabled")
        else:
            self._publish_status("enabled")
            self.get_logger().info("Chase controller enabled")
    
    def _mode_callback(self, msg: String):
        """Handle chase mode commands."""
        if msg.data in ["chase", "deterrent"]:
            self._chase_mode = msg.data
            self._reset_controllers()
            self.get_logger().info(f"Chase mode set to: {self._chase_mode}")
        else:
            self.get_logger().warn(f"Invalid chase mode: {msg.data}")
    
    def _get_robot_pose(self) -> bool:
        """Get the current pose of the robot using TF lookup (similar to pa3)."""
        try:
            # Process any pending callbacks
            rclpy.spin_once(self, timeout_sec=0.01)
            
            # For lookup_transform, get robot position in map frame
            tf_msg = self.tf_buffer.lookup_transform(
                'map',                # Target frame 
                'rosbot/base_link',   # Source frame (pa3 style)
                rclpy.time.Time(),    # Get latest transform
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Extract position - this is the position of base_link in the map frame
            self._robot_x = tf_msg.transform.translation.x
            self._robot_y = tf_msg.transform.translation.y
            
            # Extract rotation (yaw) from quaternion
            q = [
                tf_msg.transform.rotation.x,
                tf_msg.transform.rotation.y,
                tf_msg.transform.rotation.z,
                tf_msg.transform.rotation.w
            ]
            _, _, self._robot_theta = tf_transformations.euler_from_quaternion(q)
            
            return True
            
        except TransformException as ex:
            self.get_logger().debug(f"Could not get transform: {ex}")
            return False
    
    def _target_callback(self, msg: Pose):
        """Handle target pose messages from perception node."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        self._target_pose = msg
        self._last_target_time = current_time
        
        self._publish_status("target_detected")
    
    def _laser_callback(self, msg: LaserScan):
        """Handle laser scan messages for obstacle detection."""
        # Check for obstacles in front of robot
        front_ranges = []
        for i, range_val in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            # Check angles in front of robot using FRONT_SCAN_ANGLE
            if -FRONT_SCAN_ANGLE <= angle <= FRONT_SCAN_ANGLE:
                if msg.range_min <= range_val <= msg.range_max:
                    front_ranges.append(range_val)
        
        if front_ranges:
            min_distance = min(front_ranges)
            self._obstacle_detected = min_distance < MIN_OBSTACLE_DISTANCE
        else:
            self._obstacle_detected = False
    
    def _control_loop(self):
        """Main control loop implementing chase behavior."""
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Update robot pose using TF2 (similar to pa3)
        pose_available = self._get_robot_pose()
        
        # Publish robot transform (still needed for other systems)
        self._publish_transform()
        
        # Check for target timeout
        if self._check_target_timeout(current_time):
            self._publish_status("target_lost")
            self._publish_state("IDLE")
            self._publish_fsm_status("target_lost")
        
        # Execute chase behavior if enabled
        if self._chase_enabled and self._target_pose is not None:
            if pose_available:  # Only proceed if we have valid robot pose
                # Calculate distance to target for state transitions
                target_distance = self._calculate_target_distance()
                
                # Automatic state transitions based on distance (only chase -> deterrent)
                self._update_chase_mode_based_on_distance(target_distance)
                
                if self._chase_mode == "chase":
                    self._direct_chase_behavior(current_time)
                    self._publish_state("CHASE")
                elif self._chase_mode == "deterrent":
                    self._deterrent_behavior(current_time)
                    self._publish_state("DETERRENT")
                    
                # Publish FSM status with distance information (only if still enabled)
                if self._chase_enabled:
                    self._publish_fsm_status(f"active_distance_{target_distance:.2f}m")
            else:
                self.get_logger().debug("Waiting for robot pose from TF2...")
                self._publish_state("WAITING")
                self._publish_fsm_status("waiting_for_pose")
        else:
            # Stop robot if not chasing or if deterrent is complete
            self._stop_robot()
            if not self._chase_enabled and self._test_running == False:
                self._publish_state("STOPPED")
                self._publish_fsm_status("stopped")
            else:
                self._publish_state("IDLE")
                self._publish_fsm_status("disabled")
    
    def _calculate_target_distance(self) -> float:
        """Calculate distance to current target."""
        if self._target_pose is None:
            return float('inf')
        
        target_x = self._target_pose.position.x
        target_y = self._target_pose.position.y
        
        # Calculate distance from robot to target
        dx = target_x - self._robot_x
        dy = target_y - self._robot_y
        return math.sqrt(dx**2 + dy**2)
    
    def _update_chase_mode_based_on_distance(self, target_distance: float):
        """Automatically update chase mode based on distance to target."""
        previous_mode = self._chase_mode
        
        if self._chase_mode == "chase":
            # Switch to deterrent when close enough
            if target_distance <= CHASE_TO_DETERRENT_DISTANCE:
                self._chase_mode = "deterrent"
                self.get_logger().info(f"Auto-switching to DETERRENT mode (distance: {target_distance:.2f}m)")
        
        # Note: No return to chase state - deterrent is final state
        
        # Reset controllers if mode changed
        if previous_mode != self._chase_mode:
            self._reset_controllers()
    
    def _publish_state(self, state: str):
        """Publish current state for external FSM."""
        state_msg = String()
        state_msg.data = state
        self._state_pub.publish(state_msg)
    
    def _publish_fsm_status(self, status: str):
        """Publish FSM status information."""
        fsm_msg = String()
        fsm_msg.data = status
        self._fsm_status_pub.publish(fsm_msg)
    
    def _direct_chase_behavior(self, current_time: float):
        """Direct chase behavior using PID control with lidar obstacle avoidance."""
        if self._target_pose is None:
            return
        
        # Calculate distance and angle to target
        target_x = self._target_pose.position.x
        target_y = self._target_pose.position.y
        
        # Transform target from robot frame to world frame
        dx = target_x - self._robot_x
        dy = target_y - self._robot_y
        target_distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        
        # Distance error (positive means too far, negative means too close)
        distance_error = target_distance - TARGET_DISTANCE
        
        # Angular error (angle to target relative to robot heading)
        angular_error = target_angle - self._robot_theta
        
        # Normalize angular error to [-pi, pi]
        while angular_error > math.pi:
            angular_error -= 2 * math.pi
        while angular_error < -math.pi:
            angular_error += 2 * math.pi
        
        # PID control
        linear_output = self._linear_pid.update(distance_error, current_time)
        angular_output = self._angular_pid.update(angular_error, current_time)
        
        # Apply velocity limits
        linear_velocity = max(-MAX_LINEAR_VELOCITY, 
                            min(MAX_LINEAR_VELOCITY, linear_output))
        angular_velocity = max(-MAX_ANGULAR_VELOCITY, 
                             min(MAX_ANGULAR_VELOCITY, angular_output))
        
        # Obstacle avoidance using lidar
        if self._obstacle_detected:
            if distance_error > 0:  # Only avoid if we're trying to approach
                # Stop forward motion and turn to avoid obstacle
                linear_velocity = 0.0
                # Turn away from obstacle (use sign of angular error to determine direction)
                if angular_error > 0:
                    angular_velocity = OBSTACLE_AVOIDANCE_ANGULAR_VEL
                else:
                    angular_velocity = -OBSTACLE_AVOIDANCE_ANGULAR_VEL
                
                self.get_logger().debug("Obstacle detected - avoiding")
        
        self._publish_velocity(linear_velocity, angular_velocity)
        
        # Check if target distance is achieved
        if abs(distance_error) < DISTANCE_TOLERANCE:
            self._publish_status("target_reached")
        else:
            if self._obstacle_detected:
                self._publish_status("avoiding_obstacle")
            else:
                self._publish_status("chasing_target")
    
    def _deterrent_behavior(self, current_time: float):
        """Implement deterrent behavior (play sound to scare target and stop)."""
        if self._target_pose is None:
            return
        
        # Calculate distance to target
        target_x = self._target_pose.position.x
        target_y = self._target_pose.position.y
        target_distance = math.sqrt(target_x**2 + target_y**2)
        
        # Check if robot is close enough to play deterrent sound
        if target_distance <= DETERRENT_SOUND_DISTANCE:
            # Play sound if we haven't played it recently
            if (self._last_sound_time is None or 
                (current_time - self._last_sound_time) >= SOUND_INTERVAL):
                
                self._play_deterrent_sound(current_time)
                
                # Calculate and record Time-to-Clear (TTC)
                if self._chase_start_time is not None:
                    ttc = current_time - self._chase_start_time
                    self._ttc_measurements.append(ttc)
                    
                    # Calculate average TTC
                    avg_ttc = sum(self._ttc_measurements) / len(self._ttc_measurements)
                    
                    self.get_logger().info("=" * 60)
                    self.get_logger().info("🎯 TIME-TO-CLEAR (TTC) METRICS")
                    self.get_logger().info("=" * 60)
                    self.get_logger().info(f"Test #{self._current_test_number} TTC: {ttc:.2f} seconds")
                    self.get_logger().info(f"Average TTC ({len(self._ttc_measurements)} tests): {avg_ttc:.2f} seconds")
                    self.get_logger().info(f"All TTC measurements: {[f'{t:.2f}s' for t in self._ttc_measurements]}")
                    self.get_logger().info("=" * 60)
                    
                    # Reset chase start time
                    self._chase_start_time = None
                
                # After playing sound, stop the chase controller
                self.get_logger().info("Deterrent sound played - stopping chase controller")
                self._chase_enabled = False
                self._test_running = False
                self._publish_status("deterrent_complete")
                self._publish_state("STOPPED")
                self._publish_fsm_status("deterrent_complete_stopped")
            
            self._publish_status("deterrent_active_with_sound")
        else:
            self._publish_status("deterrent_waiting")
        
        # Stop the robot - deterrent mode only plays sound, no movement
        self._stop_robot()
    
    def _play_deterrent_sound(self, current_time: float):
        """Play deterrent sound with timing control to avoid spam."""
        # Check if enough time has passed since last sound
        if (self._last_sound_time is None or 
            (current_time - self._last_sound_time) >= SOUND_INTERVAL):
            
            # Play sound directly (integrated sound player functionality)
            self._play_sound_effect()
            
            # Update sound state
            self._last_sound_time = current_time
            self._sound_playing = True
            
            self.get_logger().info("Playing deterrent sound to scare target")
        else:
            # Check if we should stop indicating sound is playing
            # (for status reporting - actual sound duration depends on external system)
            time_since_sound = current_time - self._last_sound_time
            if time_since_sound > 1.0:  # Assume sound lasts ~1 second
                self._sound_playing = False
    
    def _play_sound_effect(self):
        """Play deterrent sound effect directly (integrated sound player)."""
        # Increment sound counter
        if not hasattr(self, '_sound_count'):
            self._sound_count = 0
        self._sound_count += 1
        
        # Try to play the actual alarm sound
        sound_played_successfully = False
        
        if self._sound_initialized and self._alarm_sound_path:
            try:
                # Load and play the alarm sound
                pygame.mixer.music.load(self._alarm_sound_path)
                pygame.mixer.music.play()
                
                self.get_logger().info(f"🔊 PLAYING ALARM SOUND #{self._sound_count}")
                sound_played_successfully = True
                
            except Exception as e:
                self.get_logger().error(f"Failed to play alarm sound: {e}")
                sound_played_successfully = False
        
        # Fallback to simulated sound if real sound failed or not available
        if not sound_played_successfully:
            self._play_simulated_sound()
        
        # Publish sound command for external sound systems (optional)
        sound_msg = Empty()
        self._sound_pub.publish(sound_msg)
    
    def _play_simulated_sound(self):
        """Play simulated deterrent sound when real sound is not available."""
        if not hasattr(self, '_sound_count'):
            self._sound_count = 0
        
        # Visual and text-based sound simulation
        sound_pattern = "🚨 ALARM! ALARM! ALARM! 🚨"
        
        self.get_logger().warn("=" * 50)
        self.get_logger().warn("🔊 SIMULATED ALARM SOUND (pygame not available)")
        self.get_logger().warn(sound_pattern)
        self.get_logger().warn(f"Sound #{self._sound_count}: DETERRENT ACTIVATED!")
        self.get_logger().warn("🚨 INTRUDER DETECTED - PLAYING ALARM 🚨")
        self.get_logger().warn("=" * 50)
        
        # Additional console output for emphasis
        print("\n" + "🚨" * 20)
        print("   DETERRENT SOUND ACTIVATED!")
        print("   🔊 ALARM! ALARM! ALARM! 🔊")
        print("🚨" * 20 + "\n")
        
        self.get_logger().info(f"Simulated alarm sound #{self._sound_count} played successfully")
    
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
        
        # Store velocity commands for odometry integration
        self._last_linear_vel = linear
        self._last_angular_vel = angular
    
    def _stop_robot(self):
        """Stop the robot."""
        self._publish_velocity(0.0, 0.0)
    
    def _publish_status(self, status: str):
        """Publish chase controller status."""
        status_msg = String()
        status_msg.data = status
        self._status_pub.publish(status_msg)
    
    def _reset_controllers(self):
        """Reset PID controllers and sound state."""
        self._linear_pid.reset()
        self._angular_pid.reset()
        
        # Reset sound state
        self._last_sound_time = None
        self._sound_playing = False
    
    def _publish_transform(self):
        """Publish robot transform to /base_link."""
        transform = TransformStamped()
        
        # Header
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = DEFAULT_WORLD_FRAME
        transform.child_frame_id = DEFAULT_BASE_FRAME
        
        # Translation
        transform.transform.translation.x = self._robot_x
        transform.transform.translation.y = self._robot_y
        transform.transform.translation.z = 0.0
        
        # Rotation (convert from yaw angle to quaternion)
        quaternion = tf_transformations.quaternion_from_euler(0, 0, self._robot_theta)
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        
        # Broadcast transform
        self._tf_broadcaster.sendTransform(transform)
    
    def _calculate_distance(self, point1: Point, point2: Point) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2 + 
            (point1.z - point2.z)**2
        )
    
    def _start_testing_interface(self):
        """Start the testing interface in a separate thread."""
        self.get_logger().info("Starting integrated testing interface...")
        
        # Start command interface thread
        self._command_thread = threading.Thread(target=self._testing_command_interface, daemon=True)
        self._command_thread.start()
    
    def _testing_command_interface(self):
        """Interactive command interface for testing."""
        self.get_logger().info("=== Chase Controller Ready ===")
        self.get_logger().info("Enter target intruder pose coordinates:")
        self.get_logger().info("Commands: test <x,y>, status, ttc, help, quit")
        self.get_logger().info("Example: test 3.5,4.2")
        
        while rclpy.ok():
            try:
                command = input("\nChase Controller> ").strip().lower()
                
                if command.startswith("test "):
                    coords = command[5:].strip()
                    self._run_test_target(coords)
                elif command == "status":
                    self._show_status()
                elif command == "ttc" or command == "metrics":
                    self._show_ttc_metrics()
                elif command == "help":
                    self._show_help()
                elif command == "quit" or command == "exit":
                    self.get_logger().info("Exiting chase controller...")
                    break
                else:
                    self.get_logger().warn(f"Unknown command: {command}")
                    self.get_logger().info("Type 'help' for available commands")
                    
            except EOFError:
                break
            except Exception as e:
                self.get_logger().error(f"Command error: {e}")
    
    def _run_test_target(self, coords: str):
        """Run a test with user-defined target coordinates."""
        try:
            x, y = map(float, coords.split(','))
            
            # Always start in chase mode - will auto-transition to deterrent when close
            self._chase_mode = "chase"
            self._publish_test_target(x, y)
            self._chase_enabled = True
            self._test_running = True
            self._current_test_target = f"intruder_({x},{y})"
            
            # Record chase start time for TTC calculation
            self._chase_start_time = self.get_clock().now().nanoseconds / 1e9
            self._current_test_number += 1
            
            self.get_logger().info(f"Targeting intruder at ({x}, {y})")
            self.get_logger().info("Starting in CHASE mode - will auto-switch to DETERRENT when within 1m")
            self.get_logger().info(f"Test #{self._current_test_number} - Chase timer started")
            
        except ValueError:
            self.get_logger().error("Invalid coordinates. Use format: x,y (e.g., 2.0,3.5)")
    
    def _publish_test_target(self, x: float, y: float):
        """Publish a test target pose."""
        # Create target pose message
        target_pose = Pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = 0.0
        target_pose.orientation.w = 1.0
        
        # Update internal target
        self._target_pose = target_pose
        self._last_target_time = self.get_clock().now().nanoseconds / 1e9
    
    def _show_status(self):
        """Show current system status."""
        # Get current robot pose
        pose_available = self._get_robot_pose()
        
        self.get_logger().info("=== Chase Controller Status ===")
        self.get_logger().info(f"Chase enabled: {self._chase_enabled}")
        self.get_logger().info(f"Chase mode: {self._chase_mode}")
        self.get_logger().info(f"Test running: {self._test_running}")
        if self._test_running:
            self.get_logger().info(f"Current target: {self._current_test_target}")
        
        if pose_available:
            self.get_logger().info(f"Robot pose: ({self._robot_x:.2f}, {self._robot_y:.2f}, {self._robot_theta:.2f})")
        else:
            self.get_logger().info("Robot pose: Not available")
            
        if self._target_pose:
            target_distance = self._calculate_target_distance()
            self.get_logger().info(f"Target: ({self._target_pose.position.x:.2f}, {self._target_pose.position.y:.2f})")
            self.get_logger().info(f"Distance to target: {target_distance:.2f}m")
        else:
            self.get_logger().info("Target: None")
            
        # Show TTC information
        if self._chase_start_time is not None:
            current_time = self.get_clock().now().nanoseconds / 1e9
            elapsed_time = current_time - self._chase_start_time
            self.get_logger().info(f"Current chase elapsed time: {elapsed_time:.2f}s")
        
        if self._ttc_measurements:
            avg_ttc = sum(self._ttc_measurements) / len(self._ttc_measurements)
            self.get_logger().info(f"TTC measurements: {len(self._ttc_measurements)} tests, avg: {avg_ttc:.2f}s")
    
    def _show_ttc_metrics(self):
        """Show Time-to-Clear (TTC) metrics."""
        if self._ttc_measurements:
            self.get_logger().info("=== Time-to-Clear (TTC) Metrics ===")
            self.get_logger().info(f"All TTC measurements: {[f'{t:.2f}s' for t in self._ttc_measurements]}")
            self.get_logger().info(f"Average TTC: {sum(self._ttc_measurements) / len(self._ttc_measurements):.2f} seconds")
        else:
            self.get_logger().info("No TTC measurements available.")
    
    def _show_help(self):
        """Show help information."""
        self.get_logger().info("=== Chase Controller Commands ===")
        self.get_logger().info("test <x,y>           - Set target intruder pose and start chase")
        self.get_logger().info("status               - Show system status")
        self.get_logger().info("ttc/metrics          - Show Time-to-Clear (TTC) metrics")
        self.get_logger().info("help                 - Show this help")
        self.get_logger().info("quit/exit            - Exit program")
        self.get_logger().info("")
        self.get_logger().info("Behavior:")
        self.get_logger().info("- Robot uses direct chase with PID control")
        self.get_logger().info("- Lidar-based obstacle avoidance")
        self.get_logger().info("- Auto-switches to DETERRENT when within 1.0m of target")
        self.get_logger().info("- Plays alarm sound and STOPS after deterrent activation")
        self.get_logger().info("- Publishes states to /chase_state and /fsm_status topics")
        self.get_logger().info("- States: IDLE, WAITING, CHASE, DETERRENT, STOPPED")


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