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
from std_msgs.msg import Header
import numpy as np
import math
from collections import deque
import heapq

# Topic names

# Frequency at which the loop operates
FREQUENCY = 10 #Hz.
VELOCITY = 0.15 #m/s (reduced for better mapping)

# Velocities that will be used
NODE_NAME = "terrain_explorer"
LINEAR_VELOCITY = 0.15 # m/s
ANGULAR_VELOCITY = math.pi/6 # rad/s (reduced for smoother turns)
DEFAULT_SCAN_TOPIC = 'scan'
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
TF_BASE_LINK = 'base_link'
TF_LASER_LINK = 'laser'

# Field of view in radians that is checked in front of the robot
MIN_SCAN_ANGLE_RAD = -10.0 / 180 * math.pi;
MAX_SCAN_ANGLE_RAD = +10.0 / 180 * math.pi;

USE_SIM_TIME = True

class TerrainExplorer(Node):
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
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)

        self.grid_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)

        # Setting up publishers/subscribers.
        self._laser_sub = self.create_subscription(LaserScan, DEFAULT_SCAN_TOPIC, self._laser_callback, 1)

        # Setting up transformation listener.
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.globalgrid = None

        self.last_time = self.get_clock().now()

        # Grid parameters
        self.grid_res = 0.1  
        self.grid_width = 300      
        self.grid_height = 300     
        self.origin_x = -15.0     
        self.origin_y = -15.0

        # Initialize grid: -1 = unknown, 0 = free, 1 = occupied
        self.globalgrid = np.full((self.grid_height, self.grid_width), -1, dtype=np.float32)
        
        # Exploration state
        self.exploration_complete = False
        self.current_target = None
        self.visited_positions = set()
        self.frontiers = []
        self.patrol_path = []
        self.patrol_index = 0
        
        # Safety parameters
        self.min_obstacle_distance = 0.3  # minimum distance to obstacles
        self.exploration_radius = 8.0     # how far to explore from start
        self.frontier_min_size = 3        # minimum frontier size to consider
        
        # Timer for main exploration loop
        self.exploration_timer = self.create_timer(0.5, self.exploration_loop)
        
        # Grid publishing timer
        self.grid_timer = self.create_timer(1.0, self.grid_publisher)

    def _laser_callback(self, msg):
        """Process laser scan data and update occupancy grid."""
        current_time = self.get_clock().now()
        laser_ranges = msg.ranges
        stamp = msg.header.stamp
        self.gridbuilder(msg, stamp)

    def bresenham(self, x0, y0, x1, y1):
        """Bresenham line algorithm for ray tracing."""
        points = []  
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1 
        sy = 1 if y0 < y1 else -1  
        err = dx - dy
        cx, cy = x0, y0

        while True:
            points.append((cx, cy)) 
            if cx == x1 and cy == y1:
                break
            e2 = 2 * err  
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy
        return points  

    def get_yaw_from_quaternion(self, q):
        """Convert quaternion to yaw angle."""
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        return yaw

    def gridbuilder(self, msg, stamp):
        """Build occupancy grid from laser scan data."""
        pose = self.getpose('laser', 'odom', stamp)
        if pose[0] is None:
            return
            
        x, y, yaw = pose
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        for i, range_val in enumerate(msg.ranges):
            if msg.range_min < range_val < msg.range_max:
                angle = angle_min + i * angle_increment
                global_angle = yaw + angle
                hit_x = x + range_val * math.cos(global_angle)
                hit_y = y + range_val * math.sin(global_angle)

                # Convert to grid coordinates
                grid_x = int((hit_x - self.origin_x) / self.grid_res)
                grid_y = int((hit_y - self.origin_y) / self.grid_res)

                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    self.globalgrid[grid_y, grid_x] = 100.0  # occupied

                # Ray tracing for free space
                grid_x_robot = int((x - self.origin_x) / self.grid_res)
                grid_y_robot = int((y - self.origin_y) / self.grid_res)
                
                if (0 <= grid_x_robot < self.grid_width and 
                    0 <= grid_y_robot < self.grid_height):
                    
                    free_cells = self.bresenham(grid_x_robot, grid_y_robot, grid_x, grid_y)
                    for fx, fy in free_cells[:-1]:
                        if 0 <= fx < self.grid_width and 0 <= fy < self.grid_height:
                            if self.globalgrid[fy, fx] == -1:  # only update unknown cells
                                self.globalgrid[fy, fx] = 0.0  # free

    def find_frontiers(self):
        """Find frontier cells (boundaries between free and unknown space)."""
        frontiers = []
        
        for y in range(1, self.grid_height - 1):
            for x in range(1, self.grid_width - 1):
                if self.globalgrid[y, x] == 0.0:  # free cell
                    # Check if adjacent to unknown space
                    adjacent_unknown = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if (dx == 0 and dy == 0):
                                continue
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.grid_width and 
                                0 <= ny < self.grid_height and
                                self.globalgrid[ny, nx] == -1):
                                adjacent_unknown = True
                                break
                        if adjacent_unknown:
                            break
                    
                    if adjacent_unknown:
                        # Convert back to world coordinates
                        world_x = x * self.grid_res + self.origin_x
                        world_y = y * self.grid_res + self.origin_y
                        frontiers.append((world_x, world_y))
        
        return self.cluster_frontiers(frontiers)

    def cluster_frontiers(self, frontiers):
        """Group nearby frontier points into clusters."""
        if not frontiers:
            return []
            
        clusters = []
        visited = set()
        
        for i, frontier in enumerate(frontiers):
            if i in visited:
                continue
                
            cluster = [frontier]
            visited.add(i)
            
            # Find nearby frontiers
            for j, other_frontier in enumerate(frontiers):
                if j in visited:
                    continue
                    
                dist = math.hypot(frontier[0] - other_frontier[0], 
                                frontier[1] - other_frontier[1])
                if dist < 0.5:  # cluster radius
                    cluster.append(other_frontier)
                    visited.add(j)
            
            if len(cluster) >= self.frontier_min_size:
                # Return centroid of cluster
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                clusters.append((cx, cy))
        
        return clusters

    def is_safe_position(self, x, y):
        """Check if a position is safe (not too close to obstacles)."""
        grid_x = int((x - self.origin_x) / self.grid_res)
        grid_y = int((y - self.origin_y) / self.grid_res)
        
        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
            return False
            
        # Check area around the position
        safety_radius = int(self.min_obstacle_distance / self.grid_res)
        
        for dy in range(-safety_radius, safety_radius + 1):
            for dx in range(-safety_radius, safety_radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    if self.globalgrid[ny, nx] == 100.0:  # occupied
                        return False
        return True

    def find_next_exploration_target(self):
        """Find the next target for exploration using frontier-based approach."""
        current_pos = self.get_current_position()
        if current_pos[0] is None:
            return None
            
        robot_x, robot_y = current_pos[0], current_pos[1]
        
        # Find frontiers
        frontiers = self.find_frontiers()
        
        if not frontiers:
            self.get_logger().info("No more frontiers found - exploration complete!")
            self.exploration_complete = True
            self.generate_patrol_path()
            return None
        
        # Filter frontiers by distance and safety
        valid_frontiers = []
        for fx, fy in frontiers:
            dist = math.hypot(fx - robot_x, fy - robot_y)
            if (dist < self.exploration_radius and 
                self.is_safe_position(fx, fy)):
                valid_frontiers.append((fx, fy, dist))
        
        if not valid_frontiers:
            # If no valid frontiers, try to move to a random free space
            return self.find_random_free_space()
        
        # Choose closest valid frontier
        valid_frontiers.sort(key=lambda x: x[2])
        target_x, target_y, _ = valid_frontiers[0]
        
        self.get_logger().info(f"Next exploration target: ({target_x:.2f}, {target_y:.2f})")
        return (target_x, target_y)

    def find_random_free_space(self):
        """Find a random free space for exploration when no frontiers available."""
        free_spaces = []
        
        # Sample the grid for free spaces
        for y in range(0, self.grid_height, 10):
            for x in range(0, self.grid_width, 10):
                if self.globalgrid[y, x] == 0.0:
                    world_x = x * self.grid_res + self.origin_x
                    world_y = y * self.grid_res + self.origin_y
                    if self.is_safe_position(world_x, world_y):
                        free_spaces.append((world_x, world_y))
        
        if free_spaces:
            import random
            return random.choice(free_spaces)
        return None

    def generate_patrol_path(self):
        """Generate a patrol path covering explored areas."""
        self.get_logger().info("Generating patrol path...")
        
        # Find key positions to patrol (e.g., corners, important areas)
        patrol_points = []
        
        # Sample free spaces across the explored area
        for y in range(0, self.grid_height, 20):
            for x in range(0, self.grid_width, 20):
                if self.globalgrid[y, x] == 0.0:
                    world_x = x * self.grid_res + self.origin_x
                    world_y = y * self.grid_res + self.origin_y
                    if self.is_safe_position(world_x, world_y):
                        patrol_points.append((world_x, world_y))
        
        # Simple patrol path - could be optimized with TSP solver
        if patrol_points:
            current_pos = self.get_current_position()
            if current_pos[0] is not None:
                # Sort by distance to create a reasonable path
                robot_x, robot_y = current_pos[0], current_pos[1]
                patrol_points.sort(key=lambda p: math.hypot(p[0] - robot_x, p[1] - robot_y))
        
        self.patrol_path = patrol_points
        self.patrol_index = 0
        self.get_logger().info(f"Generated patrol path with {len(patrol_points)} points")

    def get_current_position(self):
        """Get current robot position."""
        try:
            now = self.get_clock().now()
            stamp = now.to_msg()
            return self.getpose('base_link', 'odom', stamp)
        except:
            return (None, None, None)

    def exploration_loop(self):
        """Main exploration control loop."""
        if not self.exploration_complete:
            # Exploration phase
            if self.current_target is None:
                self.current_target = self.find_next_exploration_target()
                
            if self.current_target is not None:
                # Check if we've reached the current target
                current_pos = self.get_current_position()
                if current_pos[0] is not None:
                    dist = math.hypot(current_pos[0] - self.current_target[0],
                                    current_pos[1] - self.current_target[1])
                    
                    if dist < 0.3:  # reached target
                        self.get_logger().info(f"Reached target {self.current_target}")
                        self.current_target = None
                    else:
                        # Move towards target
                        stamp = self.get_clock().now().to_msg()
                        self.porter(self.current_target[0], self.current_target[1], stamp)
        else:
            # Patrol phase
            if self.patrol_path:
                if self.current_target is None:
                    self.current_target = self.patrol_path[self.patrol_index]
                    self.get_logger().info(f"Patrolling to point {self.patrol_index}: {self.current_target}")
                
                current_pos = self.get_current_position()
                if current_pos[0] is not None:
                    dist = math.hypot(current_pos[0] - self.current_target[0],
                                    current_pos[1] - self.current_target[1])
                    
                    if dist < 0.3:  # reached patrol point
                        self.patrol_index = (self.patrol_index + 1) % len(self.patrol_path)
                        self.current_target = None
                    else:
                        # Move towards patrol point
                        stamp = self.get_clock().now().to_msg()
                        self.porter(self.current_target[0], self.current_target[1], stamp)

    def stop(self):
        """Stop the robot."""
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self._cmd_pub.publish(stop_msg)

    def move_forward(self, duration):
        """Move forward for a given duration."""
        twist_msg = Twist()
        twist_msg.linear.x = self.linear_velocity
        duration = Duration(seconds=duration)
        rclpy.spin_once(self)
        start_time = self.get_clock().now()

        while rclpy.ok():
            rclpy.spin_once(self)
            if self.get_clock().now() - start_time >= duration:
                break
            self._cmd_pub.publish(twist_msg)

        self.stop()
    
    def spin(self, angle):
        """Rotate the robot by the given angle."""
        twist_msg = Twist()
        angular_speed = 0.5  # rad/s
        twist_msg.angular.z = angular_speed if angle > 0 else -angular_speed
        duration = abs(angle) / angular_speed * 1e9  
        start_time = self.get_clock().now().nanoseconds

        while rclpy.ok() and self.get_clock().now().nanoseconds - start_time < duration:
            self._cmd_pub.publish(twist_msg)
            rclpy.spin_once(self)
        
        twist_msg.angular.z = 0.0
        self._cmd_pub.publish(twist_msg)
        rclpy.spin_once(self)

    def grid_publisher(self):
        """Publish the occupancy grid."""
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"

        grid_msg.info.resolution = self.grid_res
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height

        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0 

        # Convert grid to ROS format (-1=unknown, 0=free, 100=occupied)
        ros_grid = np.copy(self.globalgrid)
        ros_grid[ros_grid == -1] = -1   # unknown
        ros_grid[ros_grid == 0.0] = 0   # free
        ros_grid[ros_grid == 100.0] = 100  # occupied
        
        flat_grid = ros_grid.astype(np.int8).flatten()
        grid_msg.data = flat_grid.tolist()
        
        self.grid_pub.publish(grid_msg)

    def porter(self, xpt, ypt, stamp):
        """Drive robot to a destination coordinate."""
        try:
            now = stamp
            while not self.tf_buffer.can_transform('odom', 'base_link', now):
                rclpy.spin_once(self, timeout_sec=0.1)
                
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', now)

            target_point = PointStamped()
            target_point.header.frame_id = 'odom'
            target_point.header.stamp = now.to_msg()
            target_point.point.x = xpt
            target_point.point.y = ypt
            target_point.point.z = 0.0

            transform = self.tf_buffer.lookup_transform('base_link', 'odom', now)
            local_target = do_transform_point(target_point, transform)

            dx = local_target.point.x
            dy = local_target.point.y
            distance = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)

            twist = Twist()

            if distance > 0.05:
                # Simple proportional control
                twist.angular.z = max(-self.angular_velocity, 
                                    min(self.angular_velocity, 2.0 * angle))
                
                if abs(angle) < 0.3:  # If roughly aligned, move forward
                    twist.linear.x = min(self.linear_velocity, distance)
                else:
                    twist.linear.x = 0.1  # Move slowly while turning
                    
                self._cmd_pub.publish(twist)
            else:
                self.stop()
                
        except Exception as e:
            self.get_logger().warn(f"Navigation error: {e}")
            rclpy.spin_once(self, timeout_sec=0.1)

    def getpose(self, target_frame, source_frame, stamp):
        """Get pose transformation between frames."""
        try:
            now = stamp
            while not self.tf_buffer.can_transform(target_frame, source_frame, now):
                rclpy.spin_once(self, timeout_sec=0.1)
                
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, now)
            
            world_x = transform.transform.translation.x
            world_y = transform.transform.translation.y
            yaw = self.get_yaw_from_quaternion(transform.transform.rotation)
            
            return world_x, world_y, yaw
            
        except Exception as e:
            self.get_logger().warn(f"Could not get pose from {target_frame} to {source_frame}: {e}")
            return None, None, None

def main(args=None):
    rclpy.init(args=args)
    explorer = TerrainExplorer()
    
    try:
        explorer.get_logger().info("Starting terrain exploration...")
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        explorer.get_logger().info("Exploration interrupted by user")
    finally:
        explorer.stop()
        explorer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()