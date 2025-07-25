# leader_position_server_binary.py
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
                    if not request or request != b"ACK":
                        print(f"[Leader] unexpected request or connection lost: {request}")
                        break

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
                leader.write("Torque_Enable", TorqueMode.DISABLED.value)
                leader.disconnect()
                print(f"[Leader] follower {addr} disconnected")

if __name__ == "__main__":
    serve_forever()
