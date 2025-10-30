# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 12:22:11 2025

@author: aadi
"""

import socket
import time
import struct
import threading

from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

from lerobot.common.robot_devices.robots.utils import get_arm_id
import json

current_position = None
position_lock = threading.Lock()



def activate_calibration(leader_arms, calibration_dir):
    """After calibration all motors function in human interpretable ranges.
    Rotations are expressed in degrees in nominal range of [-180, 180],
    and linear motions (like gripper of Aloha) in nominal range of [0, 100].
    """

    def load_or_run_calibration_(name, arm, arm_type):
        arm_id = get_arm_id(name, arm_type)
        arm_calib_path = calibration_dir / f"{arm_id}.json"

        if arm_calib_path.exists():
            with open(arm_calib_path) as f:
                calibration = json.load(f)
        else:
            print("No calibration file, please complete calibration first")

        return calibration

    for name, arm in leader_arms.items():
        calibration = load_or_run_calibration_(name, arm, "leader")
        arm.set_calibration(calibration)

def update_position_loop(leader: DynamixelMotorsBus):
    global current_position
    while True:
        try:
            pos = leader.read("Present_Position")
            with position_lock:
                current_position = pos
            time.sleep(0.01)  # Read at ~100 Hz
        except Exception as e:
            print(f"[Thread] Failed to read position: {e}")
            time.sleep(0.1)

def serve_forever():
    global current_position

    # 1) Set up the Dynamixel “leader”
    calibration_dir: str = ".cache/calibration/koch"
    leader_config = DynamixelMotorsBusConfig(
        port="/dev/ttyACM0",
        motors={
            "shoulder_pan":  (1, "xl330-m077"),
            "shoulder_lift": (2, "xl330-m077"),
            "elbow_flex":    (3, "xl330-m077"),
            "wrist_flex":    (4, "xl330-m077"),
            "wrist_roll":    (5, "xl330-m077"),
            "gripper":       (6, "xl330-m077"),
        },
    )
    leader = DynamixelMotorsBus(leader_config)
    activate_calibration(leader, calibration_dir)
    leader.connect()
    leader.write("Torque_Enable", 1, "gripper")
     

    # 2) Start background thread to keep updating `current_position`
    threading.Thread(target=update_position_loop, args=(leader,), daemon=True).start()

    # 3) TCP server setup
    HOST, PORT = "0.0.0.0", 50002
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"[Leader] listening on {HOST}:{PORT}…")

        while True:
            print("[Leader] waiting for follower...")
            conn, addr = server.accept()
            print(f"[Leader] follower connected: {addr}")

            try:
                while True:
                    # wait for request
                    request = conn.recv(3)
                    
                    # respond with latest position
                    with position_lock:
                        if current_position is None:
                            continue  # nothing to send yet
                        data = struct.pack("!6i", *current_position.tolist())
                    conn.sendall(data)
            except Exception as e:
                print(f"[Leader] follower {addr} error: {e}")
            finally:
                conn.close()
                threading.Thread(target=update_position_loop, args=(leader,), daemon=True).join()
                leader.write("Torque_Enable", TorqueMode.DISABLED.value)
                leader.disconnect()
                print(f"[Leader] follower {addr} disconnected")
                serve_forever()

if __name__ == "__main__":
    serve_forever()
