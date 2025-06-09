# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:25:42 2025

@author: aadi
"""

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait
import time
import torch

from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import (
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    RevobotMotorsBusConfig,
    MotorsBusConfig,
)
from lerobot.common.robot_devices.robots.configs import (
    AlohaRobotConfig,
    KochBimanualRobotConfig,
    KochRobotConfig,
    LeKiwiRobotConfig,
    ManipulatorRobotConfig,
    MossRobotConfig,
    RobotConfig,
    So100RobotConfig,
    StretchRobotConfig,
    RevobotRobotConfig,
)


from lerobot.common.robot_devices.robots.revobot_manipulator import RevobotManipulatorRobot

calibration_dir: str = ".cache/calibration/koch"
# `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
# Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
# the number of motors in your follower arms.
max_relative_target: int | None = None

use_revobot_leader: bool = False
use_revobot_follower: bool = True
leader_robot_type: str = 'koch' 

leader_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/ttyACM0",
            motors={
                # name: (index, model)
                "shoulder_pan": [1, "xl330-m077"],
                "shoulder_lift": [2, "xl330-m077"],
                "elbow_flex": [3, "xl330-m077"],
                "wrist_flex": [4, "xl330-m077"],
                "wrist_roll": [5, "xl330-m077"],
                "gripper": [6, "xl330-m077"],
            },
        ),
    }


follower_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/ttyACM1",
            motors={
                # name: (index, model)
                "shoulder_pan": [1, "xl430-w250"],
                "shoulder_lift": [2, "xl430-w250"],
                "elbow_flex": [3, "xl330-m288"],
                "wrist_flex": [4, "xl330-m288"],
                "wrist_roll": [5, "xl330-m288"],
                "gripper": [6, "xl330-m288"],
            },
        ),
    }


cameras = {
        "phone": OpenCVCameraConfig(
            camera_index=2,
            fps=30,
            width=640,
            height=480,
        ),
        "wrist_1": OpenCVCameraConfig(
            camera_index=8,
            fps=30,
            width=640,
            height=480,
        ),
        
        "wrist_2": OpenCVCameraConfig(
            camera_index=10,
            fps=30,
            width=640,
            height=480,
        ),
        
        "wrist_3": OpenCVCameraConfig(
            camera_index=0,
            fps=30,
            width=640,
            height=480,
        ),
    }


revobot_leader_arms = {
        "main": RevobotMotorsBusConfig(
            socket_ip= "192.168.0.142",
            #socket_ip: "97.188.81.36"
            socket_port= 50001,
            motors={
                # name: (index, model)
                "shoulder_pan": [1, "a"],
                "shoulder_lift": [2, "b"],
                "elbow_flex": [3, "c"],
                "wrist_flex": [4, "d"],
                "wrist_roll": [5, "e"],
                "gripper": [6, "f"],
            },
        ),
    }


revobot_follower_arms = {
        "main": RevobotMotorsBusConfig(
            socket_ip= "192.168.0.142",
            #socket_ip: "97.188.81.36"
            socket_port= 50000,
            motors={
                # name: (index, model)
                "shoulder_pan": [1, "a"],
                "shoulder_lift": [2, "b"],
                "elbow_flex": [3, "c"],
                "wrist_flex": [4, "d"],
                "wrist_roll": [5, "e"],
                "gripper": [6, "f"],
            },
        ),
    }



# ~ Koch specific settings ~
# Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
# to squeeze the gripper and have it spring back to an open position on its own.
gripper_open_degree: float = 35.156

mock: bool = False



robot_config = RevobotRobotConfig(leader_arms=leader_arms, follower_arms=follower_arms, revobot_follower_arms=revobot_follower_arms,
                                  cameras = cameras, use_revobot_follower = use_revobot_follower, use_revobot_leader = use_revobot_leader,
                                  )
robot = RevobotManipulatorRobot(robot_config)
robot.connect()



inference_time_s = 60
fps = 30
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "/home/revolabs/aditya/aditya_lerobot_v2/outputs/train/test_policy/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    print(action)
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)