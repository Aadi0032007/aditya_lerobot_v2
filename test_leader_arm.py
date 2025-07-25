# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 11:04:54 2025

@author: aadi
"""

from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

leader_config = DynamixelMotorsBusConfig(
    port="/dev/ttyACM0",
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)



leader_arm = DynamixelMotorsBus(leader_config)

leader_arm.connect()
leader_pos = leader_arm.read("Present_Position")
print(leader_pos)
leader_arm.disconnect()

