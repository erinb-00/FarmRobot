"""
Task Manager for Predator Detection and Response System
This script implements a finite state machine (FSM) to manage the states of the system:
PATROL: Regular monitoring for predators
CHASE: Actively pursuing a detected predator
DETER: Engaging deterrent measures if the predator is too close
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

from perception import PredatorClassifierNode
from chase_controller import ChaseController
from intruder import Intruder

class TaskManager(Node):
    def __init__(self):
        super().__init__('task_manager')

        # FSM States
        self.states = ['STOP', 'PATROL', 'CHASE', 'DETER']
        self.state = 'PATROL'

        # Instantiate subsystems
        self.perception_node = PredatorClassifierNode()
        self.chase_controller = ChaseController()
        self.deterrent_system = Intruder()

        # Start spinning the other nodes in the background
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.perception_node)
        self.executor.add_node(self.chase_controller)
        self.executor.add_node(self.deterrent_system)

        # Start the executor in a background thread
        import threading
        threading.Thread(target=self.executor.spin, daemon=True).start()

        # Timer to update state machine periodically
        self.timer = self.create_timer(1.0, self.state_machine_callback)  # 1Hz loop

        self.get_logger().info("Task Manager started in PATROL state.")

    def state_machine_callback(self):
        # Run perception and get result
        is_predator = self.perception_node.image_callback_result  # should be set by perception.py

        if is_predator is None:
            self.get_logger().warn("No image classification result received yet.")
            return

        # FSM transitions
        if self.state == 'PATROL':
            if is_predator:
                self.state = 'CHASE'
                self.get_logger().info("Predator detected. Switching to CHASE.")

        elif self.state == 'CHASE':
            if not is_predator:
                self.state = 'PATROL'
                self.get_logger().info("No predator found. Returning to PATROL.")
            elif self.chase_controller.predator_is_close():
                self.state = 'DETER'
                self.get_logger().info("Predator too close. Switching to DETER.")

        elif self.state == 'DETER':
            if not is_predator:
                self.state = 'PATROL'
                self.get_logger().info("Predator gone. Returning to PATROL.")

        # Run action based on current state
        if self.state == 'PATROL':
            self.chase_controller.patrol()

        elif self.state == 'CHASE':
            self.chase_controller.chase()

        elif self.state == 'DETER':
            self.deterrent_system.activate()

def main(args=None):
    rclpy.init(args=args)
    task_manager = TaskManager()
    try:
        rclpy.spin(task_manager)
    except KeyboardInterrupt:
        pass
    finally:
        task_manager.destroy_node()
        task_manager.executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
