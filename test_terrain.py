#!/usr/bin/env python3
"""
Terrain Explorer Logic Flow - Python Implementation
This code demonstrates the logic flow from the original ROS2 terrain explorer
"""

import math
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import random

class ExplorationState(Enum):
    INITIALIZING = "initializing"
    EXPLORING = "exploring"
    PATROLLING = "patrolling"
    STOPPED = "stopped"

@dataclass
class RobotPose:
    x: float
    y: float
    yaw: float

@dataclass
class LaserReading:
    ranges: List[float]
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float

@dataclass
class GridParams:
    resolution: float = 0.1
    width: int = 300
    height: int = 300
    origin_x: float = -15.0
    origin_y: float = -15.0

class TerrainExplorerLogic:
    """Simplified logic implementation of the terrain explorer"""
    
    def __init__(self):
        # Initialize system parameters
        self.state = ExplorationState.INITIALIZING
        self.grid_params = GridParams()
        
        # Initialize occupancy grid: -1=unknown, 0=free, 100=occupied
        self.occupancy_grid = np.full(
            (self.grid_params.height, self.grid_params.width), 
            -1, dtype=np.float32
        )
        
        # Exploration state variables
        self.current_target: Optional[Tuple[float, float]] = None
        self.visited_positions: Set[Tuple[int, int]] = set()
        self.frontiers: List[Tuple[float, float]] = []
        self.patrol_path: List[Tuple[float, float]] = []
        self.patrol_index: int = 0
        
        # Robot state
        self.robot_pose = RobotPose(0.0, 0.0, 0.0)
        
        # Safety and exploration parameters
        self.min_obstacle_distance = 0.3
        self.exploration_radius = 8.0
        self.frontier_min_size = 3
        self.target_tolerance = 0.3
        
        print("🤖 Terrain Explorer initialized")
        self.state = ExplorationState.EXPLORING

    def main_loop(self):
        """Main exploration control loop - simulates the timer-based execution"""
        print("\n🚀 Starting terrain exploration...")
        
        # Simulate running for a certain number of iterations
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations and self.state != ExplorationState.STOPPED:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Simulate laser scan processing
            self.simulate_laser_scan()
            
            # Main exploration logic
            if self.state == ExplorationState.EXPLORING:
                self.exploration_phase()
            elif self.state == ExplorationState.PATROLLING:
                self.patrol_phase()
            
            # Simulate robot movement
            self.simulate_robot_movement()
            
            # Publish grid (simulate)
            if iteration % 2 == 0:  # Every 2nd iteration
                self.publish_occupancy_grid()
            
            # Small delay to simulate real-time execution
            time.sleep(0.1)
        
        print("\n🏁 Exploration simulation completed")

    def simulate_laser_scan(self):
        """Simulate laser scan data processing"""
        # Create simulated laser data
        laser_data = LaserReading(
            ranges=[random.uniform(0.5, 5.0) for _ in range(360)],
            angle_min=-math.pi,
            angle_increment=math.pi/180,
            range_min=0.1,
            range_max=10.0
        )
        
        # Process laser data
        self.process_laser_scan(laser_data)

    def process_laser_scan(self, laser_data: LaserReading):
        """Process laser scan and update occupancy grid"""
        print("📡 Processing laser scan data...")
        
        robot_x, robot_y, robot_yaw = self.robot_pose.x, self.robot_pose.y, self.robot_pose.yaw
        
        for i, range_val in enumerate(laser_data.ranges):
            if laser_data.range_min < range_val < laser_data.range_max:
                # Calculate hit point
                angle = laser_data.angle_min + i * laser_data.angle_increment
                global_angle = robot_yaw + angle
                hit_x = robot_x + range_val * math.cos(global_angle)
                hit_y = robot_y + range_val * math.sin(global_angle)
                
                # Convert to grid coordinates
                grid_x, grid_y = self.world_to_grid(hit_x, hit_y)
                
                if self.is_valid_grid_point(grid_x, grid_y):
                    # Mark as occupied
                    self.occupancy_grid[grid_y, grid_x] = 100.0
                    
                    # Ray tracing for free space
                    robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
                    if self.is_valid_grid_point(robot_grid_x, robot_grid_y):
                        free_cells = self.bresenham_line(
                            robot_grid_x, robot_grid_y, grid_x, grid_y
                        )
                        
                        # Mark free cells (except the hit point)
                        for fx, fy in free_cells[:-1]:
                            if (self.is_valid_grid_point(fx, fy) and 
                                self.occupancy_grid[fy, fx] == -1):
                                self.occupancy_grid[fy, fx] = 0.0

    def exploration_phase(self):
        """Handle exploration logic"""
        print("🔍 Exploration phase active")
        
        if self.current_target is None:
            print("🎯 Finding next exploration target...")
            self.current_target = self.find_next_exploration_target()
            
            if self.current_target is None:
                print("✅ No more exploration targets - switching to patrol mode")
                self.state = ExplorationState.PATROLLING
                self.generate_patrol_path()
                return
        
        # Check if target is reached
        if self.is_target_reached(self.current_target):
            print(f"🎯 Reached exploration target: {self.current_target}")
            self.current_target = None
        else:
            print(f"➡️  Moving to exploration target: {self.current_target}")
            self.navigate_to_target(self.current_target)

    def patrol_phase(self):
        """Handle patrol logic"""
        print("🚁 Patrol phase active")
        
        if not self.patrol_path:
            print("❌ No patrol path available")
            self.state = ExplorationState.STOPPED
            return
        
        if self.current_target is None:
            self.current_target = self.patrol_path[self.patrol_index]
            print(f"🎯 Patrolling to point {self.patrol_index}: {self.current_target}")
        
        # Check if patrol point is reached
        if self.is_target_reached(self.current_target):
            print(f"✅ Reached patrol point: {self.current_target}")
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_path)
            self.current_target = None
        else:
            print(f"➡️  Moving to patrol point: {self.current_target}")
            self.navigate_to_target(self.current_target)

    def find_next_exploration_target(self) -> Optional[Tuple[float, float]]:
        """Find the next exploration target using frontier-based approach"""
        
        # Step 1: Find frontiers
        print("🔍 Finding frontiers...")
        frontiers = self.find_frontiers()
        print(f"Found {len(frontiers)} frontier clusters")
        
        if not frontiers:
            return None
        
        # Step 2: Filter frontiers by distance and safety
        robot_x, robot_y = self.robot_pose.x, self.robot_pose.y
        valid_frontiers = []
        
        for fx, fy in frontiers:
            distance = math.hypot(fx - robot_x, fy - robot_y)
            if (distance < self.exploration_radius and 
                self.is_safe_position(fx, fy)):
                valid_frontiers.append((fx, fy, distance))
        
        if not valid_frontiers:
            print("⚠️  No valid frontiers, trying random free space...")
            return self.find_random_free_space()
        
        # Step 3: Choose closest valid frontier
        valid_frontiers.sort(key=lambda x: x[2])  # Sort by distance
        target_x, target_y, distance = valid_frontiers[0]
        print(f"🎯 Selected frontier at ({target_x:.2f}, {target_y:.2f}), distance: {distance:.2f}")
        
        return (target_x, target_y)

    def find_frontiers(self) -> List[Tuple[float, float]]:
        """Find frontier cells (boundaries between free and unknown space)"""
        frontiers = []
        
        for y in range(1, self.grid_params.height - 1):
            for x in range(1, self.grid_params.width - 1):
                if self.occupancy_grid[y, x] == 0.0:  # Free cell
                    # Check if adjacent to unknown space
                    adjacent_unknown = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if (self.is_valid_grid_point(nx, ny) and
                                self.occupancy_grid[ny, nx] == -1):
                                adjacent_unknown = True
                                break
                        if adjacent_unknown:
                            break
                    
                    if adjacent_unknown:
                        world_x, world_y = self.grid_to_world(x, y)
                        frontiers.append((world_x, world_y))
        
        return self.cluster_frontiers(frontiers)

    def cluster_frontiers(self, frontiers: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Group nearby frontier points into clusters"""
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
                
                distance = math.hypot(
                    frontier[0] - other_frontier[0],
                    frontier[1] - other_frontier[1]
                )
                if distance < 0.5:  # Cluster radius
                    cluster.append(other_frontier)
                    visited.add(j)
            
            if len(cluster) >= self.frontier_min_size:
                # Return centroid of cluster
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                clusters.append((cx, cy))
        
        return clusters

    def find_random_free_space(self) -> Optional[Tuple[float, float]]:
        """Find a random free space for exploration"""
        free_spaces = []
        
        # Sample the grid for free spaces
        for y in range(0, self.grid_params.height, 10):
            for x in range(0, self.grid_params.width, 10):
                if self.occupancy_grid[y, x] == 0.0:
                    world_x, world_y = self.grid_to_world(x, y)
                    if self.is_safe_position(world_x, world_y):
                        free_spaces.append((world_x, world_y))
        
        return random.choice(free_spaces) if free_spaces else None

    def generate_patrol_path(self):
        """Generate a patrol path covering explored areas"""
        print("🗺️  Generating patrol path...")
        
        patrol_points = []
        
        # Sample free spaces across the explored area
        for y in range(0, self.grid_params.height, 20):
            for x in range(0, self.grid_params.width, 20):
                if self.occupancy_grid[y, x] == 0.0:
                    world_x, world_y = self.grid_to_world(x, y)
                    if self.is_safe_position(world_x, world_y):
                        patrol_points.append((world_x, world_y))
        
        # Sort by distance to create a reasonable path
        if patrol_points:
            robot_x, robot_y = self.robot_pose.x, self.robot_pose.y
            patrol_points.sort(key=lambda p: math.hypot(p[0] - robot_x, p[1] - robot_y))
        
        self.patrol_path = patrol_points
        self.patrol_index = 0
        print(f"✅ Generated patrol path with {len(patrol_points)} points")

    def is_safe_position(self, x: float, y: float) -> bool:
        """Check if a position is safe (not too close to obstacles)"""
        grid_x, grid_y = self.world_to_grid(x, y)
        
        if not self.is_valid_grid_point(grid_x, grid_y):
            return False
        
        # Check area around the position
        safety_radius = int(self.min_obstacle_distance / self.grid_params.resolution)
        
        for dy in range(-safety_radius, safety_radius + 1):
            for dx in range(-safety_radius, safety_radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if (self.is_valid_grid_point(nx, ny) and
                    self.occupancy_grid[ny, nx] == 100.0):
                    return False
        return True

    def is_target_reached(self, target: Tuple[float, float]) -> bool:
        """Check if the robot has reached the target"""
        distance = math.hypot(
            self.robot_pose.x - target[0],
            self.robot_pose.y - target[1]
        )
        return distance < self.target_tolerance

    def navigate_to_target(self, target: Tuple[float, float]):
        """Navigate robot to target (simplified simulation)"""
        dx = target[0] - self.robot_pose.x
        dy = target[1] - self.robot_pose.y
        distance = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        
        if distance > 0.05:
            # Simple proportional control simulation
            angular_velocity = 2.0 * (target_angle - self.robot_pose.yaw)
            linear_velocity = min(0.15, distance)
            
            print(f"Navigation: linear={linear_velocity:.3f}, angular={angular_velocity:.3f}")
        else:
            print("🛑 Stopping - target reached")

    def simulate_robot_movement(self):
        """Simulate robot movement (simplified)"""
        if self.current_target:
            # Move slightly towards target
            dx = self.current_target[0] - self.robot_pose.x
            dy = self.current_target[1] - self.robot_pose.y
            distance = math.hypot(dx, dy)
            
            if distance > 0.1:
                # Move 10% of the way to target each iteration
                move_factor = 0.1
                self.robot_pose.x += dx * move_factor
                self.robot_pose.y += dy * move_factor
                self.robot_pose.yaw = math.atan2(dy, dx)

    def publish_occupancy_grid(self):
        """Simulate publishing the occupancy grid"""
        free_cells = np.count_nonzero(self.occupancy_grid == 0.0)
        occupied_cells = np.count_nonzero(self.occupancy_grid == 100.0)
        unknown_cells = np.count_nonzero(self.occupancy_grid == -1)
        
        print(f"📊 Grid stats - Free: {free_cells}, Occupied: {occupied_cells}, Unknown: {unknown_cells}")

    # Utility functions
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.grid_params.origin_x) / self.grid_params.resolution)
        grid_y = int((y - self.grid_params.origin_y) / self.grid_params.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        world_x = grid_x * self.grid_params.resolution + self.grid_params.origin_x
        world_y = grid_y * self.grid_params.resolution + self.grid_params.origin_y
        return world_x, world_y

    def is_valid_grid_point(self, x: int, y: int) -> bool:
        """Check if grid coordinates are valid"""
        return (0 <= x < self.grid_params.width and 
                0 <= y < self.grid_params.height)

    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham line algorithm for ray tracing"""
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
    
#!/usr/bin/env python3
"""
Terrain Explorer Logic Diagram Visualization
Creates a visual flow diagram of the terrain exploration system using Python
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

class TerrainExplorerDiagram:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 16))
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.axis('off')
        
        # Color scheme
        self.colors = {
            'init': '#e1f5fe',      # Light blue
            'process': '#f3e5f5',   # Light purple
            'decision': '#fff3e0',  # Light orange
            'action': '#e8f5e8',    # Light green
            'data': '#fce4ec',      # Light pink
            'nav': '#e0f2f1',       # Light teal
            'timer': '#fff9c4'      # Light yellow
        }
        
    def create_box(self, x, y, width, height, text, color, shape='rect'):
        """Create a styled box with text"""
        if shape == 'diamond':
            # Create diamond shape for decisions
            diamond = patches.RegularPolygon((x + width/2, y + height/2), 4, 
                                           radius=max(width, height)/2, 
                                           orientation=np.pi/4,
                                           facecolor=color, edgecolor='black', linewidth=1)
            self.ax.add_patch(diamond)
        else:
            # Create rounded rectangle
            box = FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='black', linewidth=1)
            self.ax.add_patch(box)
        
        # Add text
        self.ax.text(x + width/2, y + height/2, text, 
                    ha='center', va='center', fontsize=8, wrap=True,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    def create_arrow(self, start_x, start_y, end_x, end_y, label='', color='black'):
        """Create an arrow between two points"""
        arrow = patches.FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                      connectionstyle="arc3,rad=0", 
                                      arrowstyle='->', 
                                      mutation_scale=15, 
                                      color=color, linewidth=1.5)
        self.ax.add_patch(arrow)
        
        # Add label if provided
        if label:
            mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
            self.ax.text(mid_x, mid_y, label, ha='center', va='center', 
                        fontsize=7, bbox=dict(boxstyle="round,pad=0.2", 
                        facecolor='white', alpha=0.9))
    
    def draw_initialization_section(self):
        """Draw the initialization flow"""
        # Start node
        self.create_box(5, 90, 12, 6, 'Start\nTerrainExplorer\nNode', 
                       self.colors['init'])
        
        # Initialize parameters
        self.create_box(25, 90, 15, 6, 'Initialize Parameters\n& Publishers/Subscribers', 
                       self.colors['init'])
        
        # Setup grid
        self.create_box(5, 80, 15, 6, 'Setup Occupancy Grid\n300x300, 0.1m resolution', 
                       self.colors['init'])
        
        # Setup timers
        self.create_box(25, 80, 15, 6, 'Start Timers\nExploration: 0.5s\nGrid Publishing: 1.0s', 
                       self.colors['init'])
        
        # Arrows
        self.create_arrow(17, 93, 25, 93)
        self.create_arrow(11, 90, 11, 86)
        self.create_arrow(32.5, 90, 32.5, 86)
    
    def draw_laser_processing_section(self):
        """Draw the laser processing flow"""
        y_base = 65
        
        # Laser callback
        self.create_box(5, y_base, 12, 5, 'Laser Scan\nCallback', 
                       self.colors['process'])
        
        # Get pose
        self.create_box(22, y_base, 12, 5, 'Get Robot Pose\nlaser → odom\ntransform', 
                       self.colors['process'])
        
        # Process readings
        self.create_box(39, y_base, 12, 5, 'Process Each\nRange Reading', 
                       self.colors['process'])
        
        # Calculate hit point
        self.create_box(5, y_base-10, 15, 6, 'Calculate Hit Point\nx = robot_x + range*cos(angle)\ny = robot_y + range*sin(angle)', 
                       self.colors['process'])
        
        # Mark occupied
        self.create_box(25, y_base-10, 12, 6, 'Mark Hit Point\nas Occupied\ngrid[y,x] = 100', 
                       self.colors['process'])
        
        # Ray tracing
        self.create_box(42, y_base-10, 12, 6, 'Ray Tracing with\nBresenham\nMark free cells = 0', 
                       self.colors['process'])
        
        # Arrows
        self.create_arrow(17, y_base+2.5, 22, y_base+2.5)
        self.create_arrow(34, y_base+2.5, 39, y_base+2.5)
        self.create_arrow(45, y_base, 45, y_base-4)
        self.create_arrow(37, y_base-7, 42, y_base-7)
        self.create_arrow(25, y_base-7, 20, y_base-7)
        self.create_arrow(12.5, y_base, 12.5, y_base-4)
    
    def draw_main_exploration_loop(self):
        """Draw the main exploration control loop"""
        y_base = 45
        
        # Exploration timer
        self.create_box(5, y_base, 15, 5, 'Exploration Timer Loop\nEvery 0.5s', 
                       self.colors['timer'])
        
        # Exploration complete decision
        self.create_box(25, y_base, 12, 5, 'Exploration\nComplete?', 
                       self.colors['decision'], 'diamond')
        
        # Current target exists decision
        self.create_box(5, y_base-12, 12, 5, 'Current Target\nExists?', 
                       self.colors['decision'], 'diamond')
        
        # Find next target
        self.create_box(22, y_base-12, 15, 5, 'Find Next\nExploration Target', 
                       self.colors['action'])
        
        # Check distance to target
        self.create_box(42, y_base-12, 12, 5, 'Check Distance\nto Target', 
                       self.colors['data'])
        
        # Reached target decision
        self.create_box(60, y_base-12, 10, 5, 'Reached\nTarget?', 
                       self.colors['decision'], 'diamond')
        
        # Clear target
        self.create_box(75, y_base-12, 10, 5, 'Clear Current\nTarget', 
                       self.colors['action'])
        
        # Navigate to target
        self.create_box(60, y_base-25, 12, 5, 'Navigate to Target\nusing porter()', 
                       self.colors['nav'])
        
        # Arrows
        self.create_arrow(20, y_base+2.5, 25, y_base+2.5)
        self.create_arrow(31, y_base, 31, y_base-7, 'No')
        self.create_arrow(25, y_base-9.5, 22, y_base-9.5)
        self.create_arrow(11, y_base-12, 11, y_base-7, 'No')
        self.create_arrow(17, y_base-9.5, 22, y_base-9.5)
        self.create_arrow(37, y_base-9.5, 42, y_base-9.5, 'Yes')
        self.create_arrow(54, y_base-9.5, 60, y_base-9.5)
        self.create_arrow(70, y_base-9.5, 75, y_base-9.5, 'Yes')
        self.create_arrow(65, y_base-12, 65, y_base-20, 'No')
    
    def draw_frontier_detection(self):
        """Draw the frontier detection flow"""
        y_base = 25
        
        # Find frontiers
        self.create_box(5, y_base, 12, 5, 'Find Frontiers\nBoundaries between\nfree & unknown', 
                       self.colors['process'])
        
        # Cluster frontiers
        self.create_box(22, y_base, 12, 5, 'Cluster Nearby\nFrontiers', 
                       self.colors['process'])
        
        # Valid frontiers decision
        self.create_box(39, y_base, 12, 5, 'Valid Frontiers\nFound?', 
                       self.colors['decision'], 'diamond')
        
        # Select closest frontier
        self.create_box(56, y_base, 12, 5, 'Select Closest\nSafe Frontier', 
                       self.colors['action'])
        
        # Find random space
        self.create_box(39, y_base-12, 15, 5, 'Find Random Free Space\nOR Mark Exploration Complete', 
                       self.colors['action'])
        
        # Set target
        self.create_box(75, y_base, 10, 5, 'Set as Current\nTarget', 
                       self.colors['action'])
        
        # Arrows
        self.create_arrow(17, y_base+2.5, 22, y_base+2.5)
        self.create_arrow(34, y_base+2.5, 39, y_base+2.5)
        self.create_arrow(51, y_base+2.5, 56, y_base+2.5, 'Yes')
        self.create_arrow(68, y_base+2.5, 75, y_base+2.5)
        self.create_arrow(45, y_base, 45, y_base-7, 'No')
        self.create_arrow(54, y_base-9.5, 75, y_base-2)
    
    def draw_patrol_section(self):
        """Draw the patrol phase flow"""
        y_base = 5
        
        # Patrol path exists
        self.create_box(45, y_base+35, 12, 5, 'Patrol Path\nExists?', 
                       self.colors['decision'], 'diamond')
        
        # Current patrol target
        self.create_box(62, y_base+35, 12, 5, 'Current Target\nExists?', 
                       self.colors['decision'], 'diamond')
        
        # Select patrol point
        self.create_box(75, y_base+25, 15, 5, 'Select Next Patrol Point\nCycle through patrol_path', 
                       self.colors['action'])
        
        # Check patrol distance
        self.create_box(75, y_base+15, 12, 5, 'Check Distance to\nPatrol Point', 
                       self.colors['data'])
        
        # Reached patrol point
        self.create_box(75, y_base+5, 12, 5, 'Reached Patrol\nPoint?', 
                       self.colors['decision'], 'diamond')
        
        # Move to next index
        self.create_box(90, y_base+5, 8, 5, 'Move to Next\nPatrol Index', 
                       self.colors['action'])
        
        # Navigate to patrol
        self.create_box(60, y_base, 12, 5, 'Navigate to\nPatrol Point', 
                       self.colors['nav'])
        
        # Generate patrol path
        self.create_box(5, y_base+35, 15, 6, 'Mark Exploration Complete\nGenerate Patrol Path', 
                       self.colors['action'])
        
        # Arrows with labels
        self.create_arrow(37, y_base+40, 45, y_base+37.5, 'Yes')
        self.create_arrow(20, y_base+37.5, 45, y_base+37.5)
        self.create_arrow(57, y_base+37.5, 62, y_base+37.5, 'Yes')
        self.create_arrow(68, y_base+35, 75, y_base+27.5, 'No')
        self.create_arrow(81, y_base+25, 81, y_base+20)
        self.create_arrow(81, y_base+15, 81, y_base+10)
        self.create_arrow(87, y_base+7.5, 90, y_base+7.5, 'Yes')
        self.create_arrow(75, y_base+7.5, 72, y_base+2.5, 'No')
    
    def draw_navigation_system(self):
        """Draw the navigation control system"""
        # Navigation details box
        nav_text = """Navigation System (porter()):
1. Transform Target to base_link Frame
2. Calculate Distance & Angle
3. Apply Proportional Control:
   - angular.z = 2.0 * angle
   - linear.x = min(velocity, distance)
4. Publish Twist Command"""
        
        self.create_box(80, 50, 18, 15, nav_text, self.colors['nav'])
        
        # Safety system box
        safety_text = """Safety Systems:
• Obstacle Distance Check
• Frontier Validation
• Path Safety Verification
• Emergency Stop Capability"""
        
        self.create_box(80, 32, 18, 12, safety_text, self.colors['action'])
    
    def draw_grid_publishing(self):
        """Draw grid publishing system"""
        # Grid publisher
        self.create_box(5, 5, 15, 8, 'Grid Publisher Timer\nEvery 1.0s\n\nConvert Internal Grid\nto ROS Format\n\nPublish OccupancyGrid\nto /occupancy_grid', 
                       self.colors['timer'])
    
    def add_title_and_legend(self):
        """Add title and color legend"""
        # Title
        self.ax.text(50, 98, 'Terrain Explorer - Logic Flow Diagram', 
                    ha='center', va='center', fontsize=16, weight='bold')
        
        # Legend
        legend_y = 85
        legend_items = [
            ('Processing', self.colors['process']),
            ('Decision', self.colors['decision']),
            ('Action', self.colors['action']),
            ('Navigation', self.colors['nav']),
            ('Timer/Publisher', self.colors['timer'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x_pos = 75 + (i % 3) * 8
            y_pos = legend_y - (i // 3) * 4
            
            # Legend box
            legend_box = FancyBboxPatch((x_pos, y_pos), 3, 2,
                                      boxstyle="round,pad=0.1",
                                      facecolor=color, edgecolor='black', linewidth=1)
            self.ax.add_patch(legend_box)
            
            # Legend text
            self.ax.text(x_pos + 1.5, y_pos + 1, label, ha='center', va='center', 
                        fontsize=7, weight='bold')
    
    def create_diagram(self):
        """Create the complete diagram"""
        print("🎨 Creating terrain explorer logic diagram...")
        
        # Draw all sections
        self.draw_laser_processing_section()
        self.draw_main_exploration_loop()
        self.draw_frontier_detection()
        self.draw_patrol_section()
        self.draw_navigation_system()
        self.draw_grid_publishing()
        self.add_title_and_legend()
        
        # Add some connecting arrows between major sections
        #self.create_arrow(32.5, 80, 32.5, 70, color='blue')  # Init to laser
        self.create_arrow(32.5, 60, 32.5, 50, color='blue')  # Laser to main loop
        
        plt.tight_layout()
        return self.fig

def create_simplified_networkx_diagram():
    """Create a simplified network diagram using NetworkX"""
    try:
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes with categories
        nodes = {
            'Start': {'category': 'init'},
            'Init_Params': {'category': 'init'},
            'Setup_Grid': {'category': 'init'},
            'Laser_Callback': {'category': 'process'},
            'Get_Pose': {'category': 'process'},
            'Update_Grid': {'category': 'process'},
            'Exploration_Loop': {'category': 'timer'},
            'Complete?': {'category': 'decision'},
            'Find_Target': {'category': 'action'},
            'Find_Frontiers': {'category': 'process'},
            'Navigate': {'category': 'nav'},
            'Patrol': {'category': 'action'},
            'Publish_Grid': {'category': 'timer'}
        }
        
        for node, attrs in nodes.items():
            G.add_node(node, **attrs)
        
        # Add edges
        edges = [
            ('Start', 'Init_Params'),
            ('Init_Params', 'Setup_Grid'),
            ('Setup_Grid', 'Exploration_Loop'),
            ('Laser_Callback', 'Get_Pose'),
            ('Get_Pose', 'Update_Grid'),
            ('Exploration_Loop', 'Complete?'),
            ('Complete?', 'Find_Target'),
            ('Complete?', 'Patrol'),
            ('Find_Target', 'Find_Frontiers'),
            ('Find_Frontiers', 'Navigate'),
            ('Navigate', 'Exploration_Loop'),
            ('Patrol', 'Exploration_Loop'),
            ('Update_Grid', 'Publish_Grid')
        ]
        
        G.add_edges_from(edges)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create colors for categories
        color_map = {
            'init': '#e1f5fe',
            'process': '#f3e5f5',
            'decision': '#fff3e0',
            'action': '#e8f5e8',
            'nav': '#e0f2f1',
            'timer': '#fff9c4'
        }
        
        node_colors = [color_map[G.nodes[node]['category']] for node in G.nodes()]
        
        # Draw the graph
        plt.figure(figsize=(14, 10))
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=3000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                linewidths=2,
                with_labels=True)
        
        plt.title('Terrain Explorer - Simplified Network Flow', fontsize=16, weight='bold')
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=cat.title()) 
                          for cat, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        return plt.gcf()
        
    except ImportError:
        print("NetworkX not available, skipping network diagram")
        return None

def main_2():
    """Main function to create and display the diagrams"""
    print("Creating Terrain Explorer Logic Visualizations")
    print("=" * 50)
    
    # Create detailed flow diagram
    diagram = TerrainExplorerDiagram()
    fig1 = diagram.create_diagram()
    
    print("Detailed flow diagram created")
    
    # Create simplified network diagram
    fig2 = create_simplified_networkx_diagram()
    if fig2:
        print("Simplified network diagram created")
    
    # Show both diagrams
    plt.show()
    
    # Option to save diagrams
    save_option = input("Save diagrams to files? (y/n): ").lower()
    if save_option == 'y':
        fig1.savefig('terrain_explorer_detailed_flow.png', dpi=300, bbox_inches='tight')
        print("📁 Detailed diagram saved as 'terrain_explorer_detailed_flow.png'")
        
        if fig2:
            fig2.savefig('terrain_explorer_network.png', dpi=300, bbox_inches='tight')
            print("Network diagram saved as 'terrain_explorer_network.png'")
    
    print("Visualization complete!")


def main():
    """Main function to run the terrain explorer logic simulation"""
    print("🌟 Terrain Explorer Logic Simulation")
    print("=" * 50)
    
    explorer = TerrainExplorerLogic()
    explorer.main_loop()
    
    print("\n🎉 Simulation completed successfully!")

if __name__ == "__main__":
    main()
    main_2()