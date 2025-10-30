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

current_position = None
position_lock = threading.Lock()


def update_position_loop(robot):
    global current_position
    while True:
        try:
            pos = robot.leader_arms["main"].read("Present_Position")
            with position_lock:
                current_position = pos
            time.sleep(0.01)  # ~100 Hz
        except Exception as e:
            print(f"[Thread] Failed to read position: {e}")
            time.sleep(0.1)


def serve_forever():
    global current_position

    # 1) Set up the Dynamixel “leader”
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

    from lerobot.common.robot_devices.robots.configs import KochRobotConfig
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

    robot_config = KochRobotConfig(
        leader_arms={"main": leader_config},
        follower_arms={},
        cameras={},
    )
    robot = ManipulatorRobot(robot_config)
    robot.connect()

    # 2) Start ONE background thread to keep updating `current_position`
    threading.Thread(
        target=update_position_loop,
        args=(robot,),
        daemon=True,
    ).start()

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
                    # wait for request (3 bytes, as you had)
                    request = conn.recv(3)
                    if not request:
                        break  # client closed

                    # respond with latest position
                    with position_lock:
                        if current_position is None:
                            # no data yet, you can send zeros or skip
                            # here I'll send zeros:
                            data = struct.pack("!6f", *([0.0] * 6))
                        else:
                            # make sure it's 6 floats
                            data = struct.pack("!6f", *current_position.tolist())
                    conn.sendall(data)

            except Exception as e:
                print(f"[Leader] follower {addr} error: {e}")
            finally:
                conn.close()
                print(f"[Leader] follower {addr} disconnected")

    # if we ever exit the with (unlikely), disconnect robot
    robot.disconnect()


if __name__ == "__main__":
    serve_forever()
