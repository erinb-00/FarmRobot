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
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, Pose, Point, TransformStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
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
DEFAULT_SCAN_TOPIC = 'scan'
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

# Path planning parameters
OCCUPANCY_THRESHOLD = 50  # Values above this are considered obstacles (0-100)
ROBOT_RADIUS = 0.15  # m - robot radius for collision checking (smaller for finer maze)
PATH_RESOLUTION = 0.05  # m - resolution for path planning (match pa3_cs81 resolution)
MAX_PLANNING_DISTANCE = 10.0  # m - maximum distance for path planning

# PID gains for linear motion (distance control)
LINEAR_KP = 1.0
LINEAR_KI = 0.1
LINEAR_KD = 0.05

# PID gains for angular motion (heading control)
ANGULAR_KP = 2.0
ANGULAR_KI = 0.1
ANGULAR_KD = 0.1

# Path following PID gains
PATH_LINEAR_KP = 0.8
PATH_LINEAR_KI = 0.05
PATH_LINEAR_KD = 0.1

PATH_ANGULAR_KP = 1.5
PATH_ANGULAR_KI = 0.05
PATH_ANGULAR_KD = 0.1

# Motion limits
MAX_LINEAR_VELOCITY = 0.5  # m/s
MAX_ANGULAR_VELOCITY = 0.7  # rad/s

# Deterrent behavior parameters
DETERRENT_SOUND_DISTANCE = 2.0  # m - distance at which to start playing sound
SOUND_INTERVAL = 2.0  # seconds - minimum interval between sound plays

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
        self._map_sub = self.create_subscription(
            OccupancyGrid, DEFAULT_OCCUPANCY_GRID_TOPIC, self._map_callback, 1)
        self._enable_sub = self.create_subscription(
            Bool, DEFAULT_CHASE_ENABLE_TOPIC, self._enable_callback, 1)
        self._mode_sub = self.create_subscription(
            String, DEFAULT_CHASE_MODE_TOPIC, self._mode_callback, 1)
        
        # Transform broadcaster for robot pose
        self._tf_broadcaster = TransformBroadcaster(self)
        
        # Chase controller state
        self._chase_enabled = False
        self._chase_mode = "chase"  # "chase" or "deterrent"
        
        # Map and environment state (similar to pa3)
        self._map: Optional[OccupancyGrid] = None
        self._map_frame_id: Optional[str] = None
        self._map_received = False
        
        # Robot pose tracking (using TF2 like pa3)
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_theta = 0.0
        self._last_cmd_time = None
        self._last_linear_vel = 0.0
        self._last_angular_vel = 0.0
        
        # Occupancy grid and path planning
        self._occupancy_grid: Optional[OccupancyGrid] = None
        self._grid_data: Optional[np.ndarray] = None
        self._current_path: List[Tuple[float, float]] = []
        self._path_index = 0
        self._path_complete = True
        
        # Target tracking
        self._target_pose: Optional[Pose] = None
        self._last_target_time: Optional[float] = None
        
        # PID controllers
        self._linear_pid = PIDController(LINEAR_KP, LINEAR_KI, LINEAR_KD)
        self._angular_pid = PIDController(ANGULAR_KP, ANGULAR_KI, ANGULAR_KD)
        
        # Path following PID controllers
        self._path_linear_pid = PIDController(PATH_LINEAR_KP, PATH_LINEAR_KI, PATH_LINEAR_KD)
        self._path_angular_pid = PIDController(PATH_ANGULAR_KP, PATH_ANGULAR_KI, PATH_ANGULAR_KD)
        
        # Sound control for deterrent behavior
        self._last_sound_time: Optional[float] = None
        self._sound_playing = False
        
        # Initialize pygame for audio playback
        self._sound_initialized = False
        self._alarm_sound_path = None
        self._init_sound_system()
        
        # Obstacle detection
        self._obstacle_detected = False
        self._min_obstacle_distance = 0.3  # m
        
        # Testing mode state
        self._testing_mode = TESTING_MODE
        self._test_running = False
        self._current_test_target = None
        
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
    
    def _map_callback(self, msg: OccupancyGrid):
        """Handle occupancy grid messages (similar to pa3)."""
        if not self._map_received:
            self.get_logger().info("Map received from pa3 maze environment")
            self._map_received = True
        
        self._map = msg
        self._occupancy_grid = msg
        self._map_frame_id = msg.header.frame_id
        
        # Convert occupancy grid data to numpy array for easier processing
        width = msg.info.width
        height = msg.info.height
        self._grid_data = np.array(msg.data).reshape((height, width))
        
        self.get_logger().debug(f"Received occupancy grid: {width}x{height}, resolution: {msg.info.resolution}m")
        
        # If in testing mode and no test is running, prompt for test selection
        if self._testing_mode and not self._test_running and self._map_received:
            self._prompt_for_test_selection()
    
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
        
        # Plan path to target when target is detected and chase is enabled
        if self._chase_enabled and self._chase_mode == "chase":
            self._plan_path_to_target()
        
        self._publish_status("target_detected")
    
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
                    self._chase_behavior_with_planning(current_time)
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
    
    def _chase_behavior_with_planning(self, current_time: float):
        """Implement chase behavior using path planning and PID control."""
        if self._target_pose is None:
            return
        
        # Check if we need to replan (target moved significantly or no current path)
        if not self._current_path or self._path_complete:
            self._plan_path_to_target()
        
        # Follow the planned path
        if self._current_path and not self._path_complete:
            self._follow_path(current_time)
            self._publish_status("following_path")
        else:
            # Fallback to direct approach if no path available
            self._direct_chase_behavior(current_time)
            self._publish_status("direct_chase")
    
    def _direct_chase_behavior(self, current_time: float):
        """Direct chase behavior (original implementation) as fallback."""
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
        
        # Check if target distance is achieved
        if abs(distance_error) < DISTANCE_TOLERANCE:
            self._publish_status("target_reached")
    
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
    
    def _plan_path_to_target(self):
        """Plan a path to the target using BFS on the occupancy grid."""
        if self._occupancy_grid is None or self._target_pose is None:
            self.get_logger().warn("Cannot plan path: missing occupancy grid or target")
            return
        
        # Convert robot and target positions to grid coordinates
        robot_grid = self._world_to_grid(self._robot_x, self._robot_y)
        target_world_x = self._target_pose.position.x
        target_world_y = self._target_pose.position.y
        
        # Calculate target approach position (TARGET_DISTANCE away from target)
        target_distance = math.sqrt(target_world_x**2 + target_world_y**2)
        if target_distance > TARGET_DISTANCE:
            # Calculate position TARGET_DISTANCE away from target
            approach_ratio = (target_distance - TARGET_DISTANCE) / target_distance
            approach_x = target_world_x * approach_ratio
            approach_y = target_world_y * approach_ratio
        else:
            # Already close enough, use current robot position
            approach_x = self._robot_x
            approach_y = self._robot_y
        
        target_grid = self._world_to_grid(approach_x, approach_y)
        
        if robot_grid is None or target_grid is None:
            self.get_logger().warn("Robot or target position outside grid bounds")
            return
        
        # Run BFS to find path
        path_grid = self._bfs_path_planning(robot_grid, target_grid)
        
        if path_grid:
            # Convert grid path back to world coordinates
            self._current_path = [self._grid_to_world(gx, gy) for gx, gy in path_grid]
            self._path_index = 0
            self._path_complete = False
            self.get_logger().info(f"Planned path with {len(self._current_path)} waypoints")
        else:
            self.get_logger().warn("No path found to target")
            self._current_path = []
            self._path_complete = True
    
    def _bfs_path_planning(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """BFS path planning on occupancy grid."""
        if self._grid_data is None:
            return []
        
        height, width = self._grid_data.shape
        
        # Check if start and goal are valid
        if not self._is_valid_cell(start[0], start[1]) or not self._is_valid_cell(goal[0], goal[1]):
            return []
        
        # BFS setup
        queue = deque([(start, [start])])
        visited = set([start])
        
        # 8-connected movement (including diagonals)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while queue:
            (current, path) = queue.popleft()
            
            # Check if we reached the goal
            if current == goal:
                return path
            
            # Explore neighbors
            for dx, dy in directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_cell = (next_x, next_y)
                
                if (next_cell not in visited and 
                    0 <= next_x < height and 0 <= next_y < width and
                    self._is_valid_cell(next_x, next_y)):
                    
                    visited.add(next_cell)
                    new_path = path + [next_cell]
                    queue.append((next_cell, new_path))
        
        return []  # No path found
    
    def _is_valid_cell(self, x: int, y: int) -> bool:
        """Check if a grid cell is valid (free space) for robot movement."""
        if self._grid_data is None:
            return False
        
        height, width = self._grid_data.shape
        
        # Check bounds
        if x < 0 or x >= height or y < 0 or y >= width:
            return False
        
        # Check if cell is free (considering robot radius)
        robot_radius_cells = int(ROBOT_RADIUS / self._occupancy_grid.info.resolution)
        
        for dx in range(-robot_radius_cells, robot_radius_cells + 1):
            for dy in range(-robot_radius_cells, robot_radius_cells + 1):
                check_x, check_y = x + dx, y + dy
                
                if (0 <= check_x < height and 0 <= check_y < width):
                    cell_value = self._grid_data[check_x, check_y]
                    # Cell is obstacle if value > threshold or unknown (-1)
                    if cell_value > OCCUPANCY_THRESHOLD or cell_value == -1:
                        return False
        
        return True
    
    def _world_to_grid(self, world_x: float, world_y: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to grid coordinates."""
        if self._occupancy_grid is None:
            return None
        
        # Transform world coordinates to grid coordinates
        origin_x = self._occupancy_grid.info.origin.position.x
        origin_y = self._occupancy_grid.info.origin.position.y
        resolution = self._occupancy_grid.info.resolution
        
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        
        # Check bounds
        if (0 <= grid_x < self._occupancy_grid.info.height and 
            0 <= grid_y < self._occupancy_grid.info.width):
            return (grid_x, grid_y)
        
        return None
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        if self._occupancy_grid is None:
            return (0.0, 0.0)
        
        origin_x = self._occupancy_grid.info.origin.position.x
        origin_y = self._occupancy_grid.info.origin.position.y
        resolution = self._occupancy_grid.info.resolution
        
        world_x = origin_x + (grid_x + 0.5) * resolution
        world_y = origin_y + (grid_y + 0.5) * resolution
        
        return (world_x, world_y)
    
    def _follow_path(self, current_time: float):
        """Follow the planned path using PID control."""
        if not self._current_path or self._path_index >= len(self._current_path):
            self._path_complete = True
            return
        
        # Get current waypoint
        waypoint_x, waypoint_y = self._current_path[self._path_index]
        
        # Calculate distance to current waypoint
        dx = waypoint_x - self._robot_x
        dy = waypoint_y - self._robot_y
        distance_to_waypoint = math.sqrt(dx**2 + dy**2)
        
        # Check if we've reached the current waypoint
        if distance_to_waypoint < PATH_RESOLUTION:
            self._path_index += 1
            if self._path_index >= len(self._current_path):
                self._path_complete = True
                self.get_logger().info("Path following complete")
                return
            else:
                # Move to next waypoint
                waypoint_x, waypoint_y = self._current_path[self._path_index]
                dx = waypoint_x - self._robot_x
                dy = waypoint_y - self._robot_y
                distance_to_waypoint = math.sqrt(dx**2 + dy**2)
        
        # Calculate desired heading to waypoint
        desired_heading = math.atan2(dy, dx)
        
        # Calculate heading error
        heading_error = desired_heading - self._robot_theta
        
        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Calculate distance to final target for speed modulation
        if self._target_pose is not None:
            target_distance = math.sqrt(
                self._target_pose.position.x**2 + self._target_pose.position.y**2)
            
            # Slow down as we approach the target
            speed_factor = min(1.0, max(0.2, (target_distance - TARGET_DISTANCE) / 2.0))
        else:
            speed_factor = 1.0
        
        # PID control for path following
        linear_output = self._path_linear_pid.update(distance_to_waypoint, current_time)
        angular_output = self._path_angular_pid.update(heading_error, current_time)
        
        # Apply speed factor and limits
        linear_velocity = max(-MAX_LINEAR_VELOCITY, 
                            min(MAX_LINEAR_VELOCITY, linear_output * speed_factor))
        angular_velocity = max(-MAX_ANGULAR_VELOCITY, 
                             min(MAX_ANGULAR_VELOCITY, angular_output))
        
        # Emergency stop if obstacle detected
        if self._obstacle_detected:
            linear_velocity = 0.0
            angular_velocity = 0.3  # Slight turn to try to avoid obstacle
        
        self._publish_velocity(linear_velocity, angular_velocity)
    
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
        self._path_linear_pid.reset()
        self._path_angular_pid.reset()
        
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
        self.get_logger().info("Waiting for map from pa3 maze environment...")
        
        # Start command interface thread
        self._command_thread = threading.Thread(target=self._testing_command_interface, daemon=True)
        self._command_thread.start()
    
    def _prompt_for_test_selection(self):
        """Prompt user for test selection once map is available."""
        if not self._map_received:
            return
            
        self.get_logger().info("=== PA3 Maze Environment Ready ===")
        self.get_logger().info("Enter target intruder pose coordinates:")
        self.get_logger().info("Commands: test <x,y>, status, help, quit")
        self.get_logger().info("Example: test 3.5,4.2")
    
    def _testing_command_interface(self):
        """Interactive command interface for testing (similar to pa3)."""
        while rclpy.ok():
            try:
                if not self._map_received:
                    time.sleep(1.0)
                    continue
                    
                command = input("\nChase Controller> ").strip().lower()
                
                if command.startswith("test "):
                    coords = command[5:].strip()
                    self._run_test_target(coords)
                elif command == "status":
                    self._show_status()
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
            
            self.get_logger().info(f"Targeting intruder at ({x}, {y})")
            self.get_logger().info("Starting in CHASE mode - will auto-switch to DETERRENT when within 1m")
            
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
        
        # Plan path if in chase mode
        if self._chase_mode == "chase":
            self._plan_path_to_target()
    
    def _show_status(self):
        """Show current system status."""
        # Get current robot pose
        pose_available = self._get_robot_pose()
        
        self.get_logger().info("=== Chase Controller Status ===")
        self.get_logger().info(f"Map received: {self._map_received}")
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
    
    def _show_help(self):
        """Show help information."""
        self.get_logger().info("=== Chase Controller Commands ===")
        self.get_logger().info("test <x,y>           - Set target intruder pose and start chase")
        self.get_logger().info("status               - Show system status")
        self.get_logger().info("help                 - Show this help")
        self.get_logger().info("quit/exit            - Exit program")
        self.get_logger().info("")
        self.get_logger().info("Behavior:")
        self.get_logger().info("- Robot starts in CHASE mode")
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