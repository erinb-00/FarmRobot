import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math
import random
from collections import deque
import time

class TerrainExplorerVisualizer:
    def __init__(self):
        # Grid parameters
        self.grid_res = 0.1  
        self.grid_width = 200      
        self.grid_height = 200     
        self.origin_x = -10.0     
        self.origin_y = -10.0
        
        # Initialize grid: -1 = unknown, 0 = free, 100 = occupied
        self.grid = np.full((self.grid_height, self.grid_width), -1, dtype=np.float32)
        
        # Robot parameters
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_radius = 0.2
        
        # Exploration state
        self.exploration_complete = False
        self.current_target = None
        self.frontiers = []
        self.patrol_path = []
        self.patrol_index = 0
        self.visited_positions = []
        
        # Safety parameters
        self.min_obstacle_distance = 0.3
        self.exploration_radius = 8.0
        self.frontier_min_size = 3
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.im = None
        
        # Generate some obstacles for simulation
        self.generate_obstacles()
        
    def generate_obstacles(self):
        """Generate some random obstacles for simulation."""
        obstacles = [
            # Walls
            [(2, 2), (2, 6)],    # vertical wall
            [(6, 3), (9, 3)],    # horizontal wall
            [(-3, -2), (-3, 2)], # vertical wall
            [(-6, 4), (-2, 4)],  # horizontal wall
            
            # Circular obstacles
            [(4, -3), 1.0],      # circle at (4, -3) with radius 1.0
            [(-5, -5), 0.8],     # circle at (-5, -5) with radius 0.8
            [(7, 7), 1.2],       # circle at (7, 7) with radius 1.2
        ]
        
        # Add line obstacles
        for obs in obstacles[:4]:
            start, end = obs
            self.add_line_obstacle(start[0], start[1], end[0], end[1])
            
        # Add circular obstacles
        for obs in obstacles[4:]:
            center, radius = obs
            self.add_circular_obstacle(center[0], center[1], radius)
    
    def add_line_obstacle(self, x1, y1, x2, y2):
        """Add a line obstacle to the grid."""
        # Convert to grid coordinates
        gx1 = int((x1 - self.origin_x) / self.grid_res)
        gy1 = int((y1 - self.origin_y) / self.grid_res)
        gx2 = int((x2 - self.origin_x) / self.grid_res)
        gy2 = int((y2 - self.origin_y) / self.grid_res)
        
        # Bresenham line algorithm
        points = self.bresenham(gx1, gy1, gx2, gy2)
        for gx, gy in points:
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                # Make the obstacle thicker
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            self.grid[ny, nx] = 100.0
    
    def add_circular_obstacle(self, cx, cy, radius):
        """Add a circular obstacle to the grid."""
        gcx = int((cx - self.origin_x) / self.grid_res)
        gcy = int((cy - self.origin_y) / self.grid_res)
        grad = int(radius / self.grid_res)
        
        for gy in range(max(0, gcy - grad), min(self.grid_height, gcy + grad + 1)):
            for gx in range(max(0, gcx - grad), min(self.grid_width, gcx + grad + 1)):
                dist = math.hypot(gx - gcx, gy - gcy) * self.grid_res
                if dist <= radius:
                    self.grid[gy, gx] = 100.0
    
    def bresenham(self, x0, y0, x1, y1):
        """Bresenham line algorithm."""
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
    
    def simulate_laser_scan(self):
        """Simulate laser scan from current robot position."""
        num_rays = 360
        max_range = 5.0
        
        for i in range(num_rays):
            angle = (i / num_rays) * 2 * math.pi
            global_angle = self.robot_yaw + angle
            
            # Cast ray
            for r in np.arange(0.1, max_range, 0.05):
                hit_x = self.robot_x + r * math.cos(global_angle)
                hit_y = self.robot_y + r * math.sin(global_angle)
                
                gx = int((hit_x - self.origin_x) / self.grid_res)
                gy = int((hit_y - self.origin_y) / self.grid_res)
                
                if not (0 <= gx < self.grid_width and 0 <= gy < self.grid_height):
                    break
                    
                if self.grid[gy, gx] == 100.0:  # Hit obstacle
                    break
                    
                # Mark free space along the ray
                robot_gx = int((self.robot_x - self.origin_x) / self.grid_res)
                robot_gy = int((self.robot_y - self.origin_y) / self.grid_res)
                
                free_cells = self.bresenham(robot_gx, robot_gy, gx, gy)
                for fx, fy in free_cells:
                    if (0 <= fx < self.grid_width and 0 <= fy < self.grid_height and 
                        self.grid[fy, fx] == -1):
                        self.grid[fy, fx] = 0.0
    
    def find_frontiers(self):
        """Find frontier cells."""
        frontiers = []
        
        for y in range(1, self.grid_height - 1):
            for x in range(1, self.grid_width - 1):
                if self.grid[y, x] == 0.0:  # free cell
                    # Check if adjacent to unknown space
                    adjacent_unknown = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.grid_width and 
                                0 <= ny < self.grid_height and
                                self.grid[ny, nx] == -1):
                                adjacent_unknown = True
                                break
                        if adjacent_unknown:
                            break
                    
                    if adjacent_unknown:
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
            
            for j, other_frontier in enumerate(frontiers):
                if j in visited:
                    continue
                    
                dist = math.hypot(frontier[0] - other_frontier[0], 
                                frontier[1] - other_frontier[1])
                if dist < 0.5:
                    cluster.append(other_frontier)
                    visited.add(j)
            
            if len(cluster) >= self.frontier_min_size:
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                clusters.append((cx, cy))
        
        return clusters
    
    def is_safe_position(self, x, y):
        """Check if position is safe."""
        gx = int((x - self.origin_x) / self.grid_res)
        gy = int((y - self.origin_y) / self.grid_res)
        
        if not (0 <= gx < self.grid_width and 0 <= gy < self.grid_height):
            return False
            
        safety_radius = int(self.min_obstacle_distance / self.grid_res)
        
        for dy in range(-safety_radius, safety_radius + 1):
            for dx in range(-safety_radius, safety_radius + 1):
                nx, ny = gx + dx, gy + dy
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    if self.grid[ny, nx] == 100.0:
                        return False
        return True
    
    def find_next_target(self):
        """Find next exploration target."""
        frontiers = self.find_frontiers()
        self.frontiers = frontiers
        
        if not frontiers:
            print("No more frontiers - exploration complete!")
            self.exploration_complete = True
            self.generate_patrol_path()
            return None
        
        # Find closest safe frontier
        valid_frontiers = []
        for fx, fy in frontiers:
            dist = math.hypot(fx - self.robot_x, fy - self.robot_y)
            if dist < self.exploration_radius and self.is_safe_position(fx, fy):
                valid_frontiers.append((fx, fy, dist))
        
        if not valid_frontiers:
            return None
        
        valid_frontiers.sort(key=lambda x: x[2])
        return (valid_frontiers[0][0], valid_frontiers[0][1])
    
    def generate_patrol_path(self):
        """Generate patrol path."""
        patrol_points = []
        
        for y in range(0, self.grid_height, 20):
            for x in range(0, self.grid_width, 20):
                if self.grid[y, x] == 0.0:
                    world_x = x * self.grid_res + self.origin_x
                    world_y = y * self.grid_res + self.origin_y
                    if self.is_safe_position(world_x, world_y):
                        patrol_points.append((world_x, world_y))
        
        self.patrol_path = patrol_points
        self.patrol_index = 0
        print(f"Generated patrol path with {len(patrol_points)} points")
    
    def move_towards_target(self, target_x, target_y, speed=0.1):
        """Move robot towards target."""
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        distance = math.hypot(dx, dy)
        
        if distance > 0.05:
            # Normalize direction and move
            move_x = (dx / distance) * speed
            move_y = (dy / distance) * speed
            
            new_x = self.robot_x + move_x
            new_y = self.robot_y + move_y
            
            # Check if new position is safe
            if self.is_safe_position(new_x, new_y):
                self.robot_x = new_x
                self.robot_y = new_y
                self.robot_yaw = math.atan2(dy, dx)
                return False  # Not reached
            else:
                # Try to find alternative path
                angle_offset = random.uniform(-math.pi/4, math.pi/4)
                alt_angle = math.atan2(dy, dx) + angle_offset
                alt_x = self.robot_x + speed * math.cos(alt_angle)
                alt_y = self.robot_y + speed * math.sin(alt_angle)
                
                if self.is_safe_position(alt_x, alt_y):
                    self.robot_x = alt_x
                    self.robot_y = alt_y
                    self.robot_yaw = alt_angle
                return False
        else:
            return True  # Reached target
    
    def update_visualization(self):
        """Update the visualization."""
        # Clear the plot
        self.ax.clear()
        
        # Create color map for grid
        display_grid = np.copy(self.grid)
        # Convert to display format: unknown=gray, free=white, occupied=black
        cmap_data = np.zeros_like(display_grid)
        cmap_data[display_grid == -1] = 0.5  # unknown = gray
        cmap_data[display_grid == 0] = 1.0   # free = white
        cmap_data[display_grid == 100] = 0.0 # occupied = black
        
        # Display grid
        extent = [self.origin_x, self.origin_x + self.grid_width * self.grid_res,
                  self.origin_y, self.origin_y + self.grid_height * self.grid_res]
        self.ax.imshow(cmap_data, cmap='gray', origin='lower', extent=extent, alpha=0.8)
        
        # Draw robot
        robot_circle = Circle((self.robot_x, self.robot_y), self.robot_radius, 
                             color='blue', alpha=0.7)
        self.ax.add_patch(robot_circle)
        
        # Draw robot heading
        head_x = self.robot_x + self.robot_radius * math.cos(self.robot_yaw)
        head_y = self.robot_y + self.robot_radius * math.sin(self.robot_yaw)
        self.ax.arrow(self.robot_x, self.robot_y, 
                     head_x - self.robot_x, head_y - self.robot_y,
                     head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # Draw frontiers
        if self.frontiers:
            fx, fy = zip(*self.frontiers)
            self.ax.scatter(fx, fy, c='green', s=30, alpha=0.7, label='Frontiers')
        
        # Draw current target
        if self.current_target:
            self.ax.scatter(self.current_target[0], self.current_target[1], 
                           c='red', s=100, marker='*', label='Current Target')
        
        # Draw patrol path
        if self.patrol_path:
            px, py = zip(*self.patrol_path)
            self.ax.plot(px, py, 'r--', alpha=0.5, label='Patrol Path')
            self.ax.scatter(px, py, c='orange', s=20, alpha=0.7)
        
        # Draw visited positions
        if self.visited_positions:
            vx, vy = zip(*self.visited_positions)
            self.ax.plot(vx, vy, 'b-', alpha=0.3, linewidth=1, label='Robot Path')
        
        self.ax.set_xlim(self.origin_x, self.origin_x + self.grid_width * self.grid_res)
        self.ax.set_ylim(self.origin_y, self.origin_y + self.grid_height * self.grid_res)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        status = "Exploring" if not self.exploration_complete else "Patrolling"
        self.ax.set_title(f'Terrain Explorer - Status: {status}')
        
        plt.draw()
        plt.pause(0.1)
    
    def exploration_step(self):
        """Single step of exploration logic."""
        # Record current position
        self.visited_positions.append((self.robot_x, self.robot_y))
        
        # Simulate laser scan
        self.simulate_laser_scan()
        
        if not self.exploration_complete:
            # Exploration phase
            if self.current_target is None:
                self.current_target = self.find_next_target()
            
            if self.current_target is not None:
                reached = self.move_towards_target(self.current_target[0], self.current_target[1])
                if reached:
                    print(f"Reached exploration target: {self.current_target}")
                    self.current_target = None
        else:
            # Patrol phase
            if self.patrol_path:
                if self.current_target is None:
                    self.current_target = self.patrol_path[self.patrol_index]
                    print(f"Patrolling to point {self.patrol_index}: {self.current_target}")
                
                reached = self.move_towards_target(self.current_target[0], self.current_target[1])
                if reached:
                    self.patrol_index = (self.patrol_index + 1) % len(self.patrol_path)
                    self.current_target = None
    
    def run_exploration(self, max_steps=1000):
        """Run the exploration simulation."""
        plt.ion()  # Interactive mode on
        
        for step in range(max_steps):
            self.exploration_step()
            self.update_visualization()
            
            if step % 50 == 0:
                print(f"Step {step}, Robot at ({self.robot_x:.2f}, {self.robot_y:.2f})")
            
            # Small delay to see the animation
            time.sleep(0.05)
            
            # Stop if we've been patrolling for a while
            if self.exploration_complete and step > 500:
                break
        
        plt.ioff()  # Interactive mode off
        plt.show()
        print("Exploration simulation completed!")

# Example usage
if __name__ == "__main__":
    explorer = TerrainExplorerVisualizer()
    explorer.run_exploration(max_steps=800)