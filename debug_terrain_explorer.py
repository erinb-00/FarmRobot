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
#import skimage.draw import bresenham
#import pybresenham
#from bresenham import bresenham
import math
import networkx as nx
from collections import defaultdict
import itertools

# Topic names

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

# Field of view in radians that is checked in front of the robot (TODO: feel free to tune)
# Note: these angles are with respect to the robot perspective, but needs to be
# converted to match how the laser is mounted.
MIN_SCAN_ANGLE_RAD = -10.0 / 180 * math.pi
MAX_SCAN_ANGLE_RAD = +10.0 / 180 * math.pi

USE_SIM_TIME = True


class GridMapper(Node):
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

        #self.publisher_ = self.create_publisher(PoseArray, 'pose_sequence', 10) #change topic
        self.grid_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)

        # 2nd. setting up publishers/subscribers.
        self._laser_sub = self.create_subscription(LaserScan, DEFAULT_SCAN_TOPIC, self._laser_callback, 1)

        # Setting up transformation listener.
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.globalgrid = None

        self.create_subscription(
            LaserScan,
            DEFAULT_SCAN_TOPIC,
            self._laser_callback,
            10
        )
        self.last_time = self.get_clock().now()

        #initializing grid
        self.grid_res = 0.1
        self.grid_width = 200
        self.grid_height = 200
        self.origin_x = -10.0
        self.origin_y = -10.0

        self.globalgrid = np.full((self.grid_height, self.grid_width), -1,dtype=np.float32)

        # CPP-related attributes
        self.graph = None
        self.cpp_path = []
        self.mapping_complete = False

    def _laser_callback(self, msg):
        self.get_logger().info("Laser callback triggered.")
        current_time = self.get_clock().now()

        laser_ranges = msg.ranges
        stamp = msg.header.stamp

        self.gridbuilder(msg,stamp)

    def bresenham(self,x0, y0, x1, y1):
        points = []
        # Calculating the differences between the end points
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        # Determining the direction of the line
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        # Initializing the error term
        err = dx - dy

        # Initializing the current point
        cx, cy = x0, y0

        # Looping through the line
        while True:
            points.append((cx, cy))
            if cx == x1 and cy == y1:
                break
            # Calculating the next pixel.
            e2 = 2 * err
            # Moving along x
            if e2 > -dy:
                err -= dy
                cx += sx
            # Moving along y
            if e2 < dx:
                err += dx
                cy += sy

        return points

    def get_yaw_from_quaternion(self, q):
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        return yaw

    def gridbuilder(self,msg,stamp):
        #x,y,yaw = self.getpose('base_link','odom',stamp)
        x,y,yaw = self.getpose('laser','odom',stamp)

        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        for i, range_val in enumerate(msg.ranges):
            if msg.range_min < range_val < msg.range_max:
                angle = angle_min + i * angle_increment
                global_angle = yaw + angle
                hit_x = x + range_val * math.cos(global_angle)
                hit_y = y + range_val * math.sin(global_angle)
                print('step 1-grid')

                # Converting to grid cell
                grid_x = int((hit_x - self.origin_x) / self.grid_res)
                grid_y = int((hit_y - self.origin_y) / self.grid_res)

                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    self.globalgrid[grid_y, grid_x] = 1.0 # marking as occupied

                print('step 2-grid')

                # Raytracing to mark free space
                grid_x_robot = int((x - self.origin_x) / self.grid_res)
                grid_y_robot = int((y - self.origin_y) / self.grid_res)
                grid_x_hit = int((hit_x - self.origin_x) / self.grid_res)
                grid_y_hit = int((hit_y - self.origin_y) / self.grid_res)

                x = int(round(x))
                y = int(round(y))
                hit_x = int(round(hit_x))
                hit_y = int(round(hit_y))
                #free_cells = self.bresenham(x, y, hit_x, hit_y) #using Bresenham line algorithm to determine free cells between obstacle and robot
                free_cells = self.bresenham(grid_x_robot, grid_y_robot, grid_x_hit, grid_y_hit)

                for fx, fy in free_cells[:-1]:
                    if 0 <= fx < self.grid_width and 0 <= fy < self.grid_height:
                        self.globalgrid[fy, fx] = 0.0 # marking as free

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
        start_time = self.get_clock().now().nanoseconds
        angular_speed = 0.5 # rad/s
        twist_msg.angular.z = angular_speed if angle > 0 else -angular_speed
        duration = abs(angle) / angular_speed * 1e9

        while rclpy.ok() and self.get_clock().now().nanoseconds - start_time < duration:
            self._cmd_pub.publish(twist_msg)
            rclpy.spin_once(self)
        twist_msg.angular.z = 0.0
        self._cmd_pub.publish(twist_msg)
        rclpy.spin_once(self)

    def grid_publisher(self):
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

        # Flatten grid and convert to int8 format
        flat_grid = (self.globalgrid * 100).astype(np.int8).flatten()
        grid_msg.data = flat_grid.tolist()
        print('publishing map')
        self.grid_pub.publish(grid_msg)

    def build_graph_from_grid(self):
        """Build a graph from the occupancy grid where edges represent free paths."""
        self.graph = nx.Graph()

        # Add nodes for free cells
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.globalgrid[y, x] == 0.0:  # free space
                    # Convert grid coordinates to world coordinates
                    world_x = self.origin_x + x * self.grid_res
                    world_y = self.origin_y + y * self.grid_res
                    self.graph.add_node((x, y), pos=(world_x, world_y))

        # Add edges between adjacent free cells
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.globalgrid[y, x] == 0.0:  # free space
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and 
                            self.globalgrid[ny, nx] == 0.0):
                            # Calculate distance (Euclidean for diagonal, 1 for orthogonal)
                            distance = math.sqrt(dx*dx + dy*dy) * self.grid_res
                            self.graph.add_edge((x, y), (nx, ny), weight=distance)

    def find_odd_degree_vertices(self):
        """Find vertices with odd degree in the graph."""
        odd_vertices = []
        for node in self.graph.nodes():
            if self.graph.degree(node) % 2 == 1:
                odd_vertices.append(node)
        return odd_vertices

    def min_weight_perfect_matching(self, odd_vertices):
        """Find minimum weight perfect matching of odd degree vertices."""
        if len(odd_vertices) == 0:
            return []
        
        # Calculate shortest paths between all pairs of odd vertices
        distances = {}
        paths = {}
        for i, u in enumerate(odd_vertices):
            for j, v in enumerate(odd_vertices):
                if i < j:
                    try:
                        path = nx.shortest_path(self.graph, u, v, weight='weight')
                        dist = nx.shortest_path_length(self.graph, u, v, weight='weight')
                        distances[(u, v)] = dist
                        paths[(u, v)] = path
                    except nx.NetworkXNoPath:
                        distances[(u, v)] = float('inf')
                        paths[(u, v)] = []

        # Find minimum weight perfect matching (simplified greedy approach)
        matching = []
        remaining = odd_vertices.copy()
        while len(remaining) > 1:
            min_dist = float('inf')
            best_pair = None
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    u, v = remaining[i], remaining[j]
                    key = (u, v) if (u, v) in distances else (v, u)
                    if distances.get(key, float('inf')) < min_dist:
                        min_dist = distances[key]
                        best_pair = (u, v)
            if best_pair:
                matching.append(best_pair)
                remaining.remove(best_pair[0])
                remaining.remove(best_pair[1])
            else:
                break

        return matching

    def solve_chinese_postman(self):
        """Solve the Chinese Postman Problem to find optimal traversal path."""
        if not self.graph or len(self.graph.nodes()) == 0:
            self.get_logger().warn("Graph is empty, cannot solve CPP")
            return []

        # Step 1: Find vertices with odd degree
        odd_vertices = self.find_odd_degree_vertices()
        self.get_logger().info(f"Found {len(odd_vertices)} odd degree vertices")

        # Step 2: If graph is already Eulerian (all vertices have even degree)
        if len(odd_vertices) == 0:
            self.get_logger().info("Graph is already Eulerian")
            # Find Eulerian circuit
            try:
                eulerian_path = list(nx.eulerian_circuit(self.graph))
                return [edge[0] for edge in eulerian_path] + [eulerian_path[-1][1]]
            except:
                self.get_logger().warn("Could not find Eulerian circuit")
                return []

        # Step 3: Find minimum weight perfect matching
        matching = self.min_weight_perfect_matching(odd_vertices)
        self.get_logger().info(f"Found matching with {len(matching)} pairs")

        # Step 4: Add matching edges to make graph Eulerian
        augmented_graph = self.graph.copy()
        for u, v in matching:
            if augmented_graph.has_edge(u, v):
                # If edge already exists, increase its multiplicity
                augmented_graph[u][v]['weight'] *= 0.5  # Reduce weight to prefer this path
            else:
                # Add the edge from shortest path
                try:
                    path = nx.shortest_path(self.graph, u, v, weight='weight')
                    for i in range(len(path) - 1):
                        if augmented_graph.has_edge(path[i], path[i+1]):
                            augmented_graph[path[i]][path[i+1]]['weight'] *= 0.8
                except:
                    continue

        # Step 5: Find Eulerian circuit in augmented graph
        try:
            eulerian_path = list(nx.eulerian_circuit(augmented_graph))
            return [edge[0] for edge in eulerian_path] + [eulerian_path[-1][1]]
        except:
            self.get_logger().warn("Could not find Eulerian circuit in augmented graph")
            return []

    def execute_cpp_path(self, stamp):
        """Execute the Chinese Postman Problem path."""
        if not self.cpp_path:
            self.get_logger().warn("No CPP path available")
            return

        self.get_logger().info(f"Executing CPP path with {len(self.cpp_path)} waypoints")

        for i, (grid_x, grid_y) in enumerate(self.cpp_path):
            # Convert grid coordinates to world coordinates
            world_x = self.origin_x + grid_x * self.grid_res
            world_y = self.origin_y + grid_y * self.grid_res

            self.get_logger().info(f"Moving to waypoint {i+1}/{len(self.cpp_path)}: ({world_x:.2f}, {world_y:.2f})")
            self.porter(world_x, world_y, stamp)
            self.grid_publisher()  # Publish updated map

        self.get_logger().info("CPP path execution completed!")

    def porter(self,xpt,ypt,stamp):
        try:
            # Getting transform from odom to base_link
            now = stamp
            while not self.tf_buffer.can_transform('odom', 'base_link', now):
                rclpy.spin_once(self,timeout_sec=0.1)
            trans = self.tf_buffer.lookup_transform('odom','base_link', now)

            # Transforming the global point to robot's local frame
            target_point = PointStamped()
            target_point.header.frame_id = 'odom'
            target_point.header.stamp = now.to_msg()
            target_point.point.x = xpt
            target_point.point.y = ypt
            target_point.point.z = 0.0

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
            while not self.tf_buffer.can_transform(target_frame, source_frame, now):
                rclpy.spin_once(self,timeout_sec=0.1)

            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, now)

            world_x = transform.transform.translation.x
            world_y = transform.transform.translation.y
            yaw = self.get_yaw_from_quaternion(transform.transform.rotation)

            return world_x, world_y,yaw

        except Exception as e:
            self.get_logger().warn(f"Could not get x, y from {target_frame} to {source_frame}: {e}")
            return None,None,None

def main(args=None):
    rclpy.init(args=args)
    stamp = rclpy.time.Time()
    gridder = GridMapper()

    # First, let the robot build the map for a specified duration
    print("Building occupancy grid map...")
    mapping_duration = float(input("Enter mapping duration in seconds (e.g., 30): "))
    start_time = gridder.get_clock().now()
    mapping_duration_obj = Duration(seconds=mapping_duration)

    # Spin for mapping phase
    while rclpy.ok() and (gridder.get_clock().now() - start_time) < mapping_duration_obj:
        rclpy.spin_once(gridder, timeout_sec=0.1)

    print("Mapping phase completed. Building graph from occupancy grid...")

    # Build graph from the occupancy grid
    gridder.build_graph_from_grid()

    if gridder.graph and len(gridder.graph.nodes()) > 0:
        print(f"Graph built with {len(gridder.graph.nodes())} nodes and {len(gridder.graph.edges())} edges")
        
        # Solve Chinese Postman Problem
        print("Solving Chinese Postman Problem...")
        gridder.cpp_path = gridder.solve_chinese_postman()
        
        if gridder.cpp_path:
            print(f"CPP solution found with {len(gridder.cpp_path)} waypoints")
            print("Executing CPP path...")
            gridder.execute_cpp_path(stamp)
        else:
            print("Could not find CPP solution")
    else:
        print("Could not build graph from occupancy grid")

    gridder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()