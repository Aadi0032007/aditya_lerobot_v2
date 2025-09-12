# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:08:13 2025

@author: aadi
"""

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.robot_devices.utils import busy_wait
import time
import torch
import numpy as np
import os

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
    MobileRevobotRobotConfig,
    MossRobotConfig,
    RobotConfig,
    So100RobotConfig,
    StretchRobotConfig,
    RevobotRobotConfig,
)


from lerobot.common.robot_devices.robots.revobot_manipulator import RevobotManipulatorRobot
from lerobot.common.robot_devices.robots.revobot_mobile_manipulator import MobileRevobotManipulator

from lerobot.common.utils.gemini_utils import bbox_2d_gemini, plot_2d_bbox
import rerun as rr

def _init_rerun(session_name: str = "Aditya_control_loop") -> None:
    
        # Configure Rerun flush batch size default to 8KB if not set
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

        # Initialize Rerun based on configuration
        rr.init(session_name)
        
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        rr.spawn(memory_limit=memory_limit)


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



robot_config = MobileRevobotRobotConfig()
robot = MobileRevobotManipulator(robot_config)



inference_time_s = 80
fps = 10
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

# ckpt_path = "/home/revolabs/aditya/aditya_lerobot_v2/outputs/train/test_2_act/checkpoints/100000/pretrained_model"
# ckpt_path = "/home/revolabs/aditya/aditya_lerobot_v2/outputs/train/test_3_long_act/checkpoints/last/pretrained_model"
# ckpt_path = "/home/revolabs/aditya/aditya_lerobot_v2/outputs/train/bartender_1_act/checkpoints/100000/pretrained_model"
# ckpt_path = "/home/revolabs/aditya/aditya_lerobot_v2/outputs/train/bartender_1_2cams_act/checkpoints/100000/pretrained_model"
# ckpt_path = "/home/revolabs/aditya/aditya_lerobot_v2/outputs/train/bartender_bb_act/checkpoints/last/pretrained_model"
ckpt_path = "/home/revolabs/aditya/aditya_lerobot_v2/outputs/train/bartender_bb_2cam_act/checkpoints/last/pretrained_model"


# policy = ACTPolicy.from_pretrained(ckpt_path)
policy = ACTPolicy.from_pretrained(ckpt_path)

policy.to(device)
robot.connect()



   
def inference(robot):
    
    start_episode_t = time.perf_counter()
    while True:
        observation, _ = robot.teleop_step(record_data=True)  
        if time.perf_counter() - start_episode_t > 3:
            break
    prompt = "Detect the 2d bounding boxes of bottle and steel-glass" 
    if observation["observation.images.phone"] is not None:
        bounding_box = bbox_2d_gemini(observation["observation.images.phone"],prompt)
        print(bounding_box)
    
    rest_position = [1.0546875, 113.115234, 172.4414, 25.136719, -0.7910156, 35.06836]
    count = 0
    # _init_rerun()
    for _ in range(inference_time_s * fps):
        start_time = time.perf_counter()
    
        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()
        image_keys = [key for key in observation if "image" in key]
        for key in image_keys:
            if  "observation.images.phone" in key:
                plot_img = plot_2d_bbox(observation[key], bounding_box)
                numpy_img = np.array(plot_img)
                observation[key] = torch.from_numpy(numpy_img)
        
        # image_keys = [key for key in observation if "image" in key]
        # for key in image_keys:
        #     rr.log(key, rr.Image(observation[key].numpy()), static=True)
    
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
        # print(action)
        # Order the robot to move
        robot.send_action(action)
    
        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)
        
        result = [a - b for a, b in zip(rest_position, action)]
        close_to_home = all(x < 1 for x in result)

        count += 1
        
        if result[1] < 5 and count > fps * 3:
            print("reaching home")
            break
        
def go_to_home_position(robot):
    """
    Moves the follower_arm from its current position to `rest_position`
    in `steps` small increments.
    """
    rest_position = [1.0546875, 113.115234, 172.4414, 25.136719, -0.7910156, 35.06836]
    current_pos = robot.follower_arms['main'].read("Present_Position")
    steps=45
    for i in range(1, steps + 1):
        alpha = i / steps
        intermediate_pos = current_pos + alpha * (rest_position - current_pos)
        robot.follower_arms['main'].write("Goal_Position", intermediate_pos)
        time.sleep(0.05)
    print("reached home position")
        
        
try:
    while True:
        print("starting inference")
        inference(robot)
        # go_to_home_position(robot)

except KeyboardInterrupt:
    print("\nCtrl+C detected. Stopping gracefully...")

finally:
    # go_to_home_position(robot)
    robot.disconnect()
    print("Robot disconnected.")

