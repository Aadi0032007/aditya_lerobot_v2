#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:20:28 2025

@author: aadi
"""

import time
import socket
import struct
from typing import List, Optional, Tuple
from collections import namedtuple
import numpy as np

from lerobot.common.robot_devices.motors.configs import RevobotMotorsBusConfig


write_call_counter = 0
pause_gripper_angle = 32
freeze_flag = 0

RobotData = namedtuple("RobotData", [
    "position",
    "delta",
    "PIDDelta",
    "forceDelta",
    "sin",
    "cos",
    "playbackPosition",
    "sentPosition",
    "joint67Data",
    "reserved"
])
# Joint67Status is 4 integers
Joint67Status = namedtuple("Joint67Status", [
    "j6Position",
    "j6Torque",
    "j7Position",
    "j7Torque"
])


class RobotInitialise:
    RD_SIZE = 40  # bytes per RobotData block (10 ints * 4 bytes)
    # Precompile struct format for 10 integers
    RD_STRUCT = struct.Struct('10i')
    
    def __init__(self,
    config: RevobotMotorsBusConfig,
    ):
        self.socket_ip = config.socket_ip
        self.socket_port = config.socket_port
        self.motors = config.motors
        self.sock = None
        self.calibration = None
        # self.skip_frames=skip_frame

        # Internal data storage for robot data.
        self.robotDataList: List[RobotData] = [
            RobotData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) for _ in range(8)
        ]
        self.joint67Status = Joint67Status(0, 0, 0, 0)
        rest_position = [ -0.43945312, 117.509766, 118.916016, 85.78125, -4.482422, 34.716797  ]

        self.temp_values = rest_position
   
    
    def create_socket(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock = sock
            return sock
        except socket.error as err:
            print("Error: socket creation failed:", err)
            return None

    def connect(self):
        # print("RevobotRobotBus.connect called")
        self.sock = self.create_socket()
        if self.sock is None:
            print("Socket creation failed.")
            return
        try:
            self.sock.connect((self.socket_ip, self.socket_port))
            print(f"Connected to Revobot at {self.socket_ip}:{self.socket_port}. Socket fd: {self.sock.fileno()}")
        except Exception as e:
            print("Error: connection with the server failed", e)
            self.sock = None

    def reconnect(self):
        # print("RevobotRobotBus.reconnect called")
        if self.sock:
            self.disconnect()
        self.connect()

    def disconnect(self):
        # print("RevobotRobotBus.disconnect called")
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                print("Error during socket close:", e)
            finally:
                self.sock = None
                print("Socket disconnected.")
        else:
            print("Socket already disconnected.")

    

    def parse_partial_robot_data_and_ignore_first(self, raw_data: bytes):
        total_len = len(raw_data)
        if total_len < self.RD_SIZE:
            print("Not enough bytes for a single RobotData block!")
            return self.robotDataList

        num_blocks = min(total_len // self.RD_SIZE, 8)  # process up to 8 blocks

        for i in range(num_blocks):
            start = i * self.RD_SIZE
            end = start + self.RD_SIZE
            block = self.RD_STRUCT.unpack(raw_data[start:end])
            self.robotDataList[i] = RobotData(*block)
            
        return self.robotDataList


    def send_command(self, command):
        if self.sock is None or self.sock.fileno() <= 0:
            print("Socket not valid")
            return -1
    
        maxRetries = 5
        bytesWritten = 0
        recvBytes = 0
    
        while recvBytes == 0 and maxRetries > 0:
            try:
                self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            except Exception:
                self.connect()  # Call connect without checking for boolean return
                print("Attempting to reconnect...")
    
            try:
                bytesWritten = self.sock.send(command.encode('utf-8'))
                # print("Sending command:", command)
            except Exception as e:
                print("send_command error:", e)
                self.reconnect()  # Try reconnecting instead of disconnecting
                maxRetries -= 1
                continue  # Retry sending command after reconnecting
    
            try:
                recv_data = self.sock.recv(1024)
                recvBytes = len(recv_data)
            except Exception:
                pass
    
            if recvBytes == 0:  # If no data received, attempt reconnect
                # print("No data received, attempting to reconnect...")
                self.reconnect()
    
            maxRetries -= 1

        if recvBytes == 0:
            print("Failed to receive data after multiple attempts.")
            return -1
            
        self.robotDataList = self.parse_partial_robot_data_and_ignore_first(recv_data)
        
        self.joint67Status = Joint67Status(
            j6Position = self.robotDataList[3].joint67Data,
            j6Torque   = self.robotDataList[4].joint67Data,
            j7Position = self.robotDataList[1].joint67Data,
            j7Torque   = self.robotDataList[2].joint67Data
        )
        
        # print(self.robotDataList)
        # print("Updated joint67Status:", self.joint67Status)
        return bytesWritten



    def read(self, data_name = "g", motor_names=None):
        # Update joint positions by sending the "get" command.
        if self.sock is None:
            print("Socket not connected; cannot read data.")
            return None
        
        n = self.send_command("xxx xxx xxx xxx g;")
        if n < 0:
            return None
        
        positions = []
        # For joints 1-5, we use playbackPosition from robotDataList.
        for joint in range(1, 6):
            if joint < len(self.robotDataList):
                positions.append(float(self.robotDataList[joint].playbackPosition)/3600)
            else:
                positions.append(0.0)
        # For joints 6 and 7, we use joint67Status.
        positions.append(float(self.joint67Status.j6Position)/88.8889)
        positions.append(float(self.joint67Status.j7Position)/120) # default 120, 171.4
        if len(positions) == 7:
            positions.pop(4)
        
        positions = np.array(positions)
        positions = positions.astype(np.float32)
        
        return positions
    

        
    def write_init(self):
        """ Add all the initialisation parameters during the socket connection
            these parameters only execute once for every socket connection."""
        
        init_config_lst = [
                "P 0 0 0 0 0",
                "S, J1BoundryHigh, 612000",
                "S, J1BoundryLow, -612000",
                "S, J2BoundryHigh, 320400",
                "S, J2BoundryLow, -320400",
                "S, J3BoundryHigh, 500400",
                "S, J3BoundryLow, -500400",
                "S, J4BoundryHigh, 400000",
                "S, J4BoundryLow, -400000",
                "S, J5BoundryHigh, 450000",
                "S, J5BoundryLow, -450000",
                "a 0 0 0 0 0",
                "a 0 0 0 36000 0",
                "a 0 0 0 36000 36000",
                "a 0 0 0 -36000 36000",
                "a 0 0 0 -36000 -36000",
                "S RebootServo 1 430 1296000 0",
                "S RebootServo 3 430 324000 0",
                "S RebootServo 4 430 1296000 0",
                "S ServoSetX 1 11 4",
                "S ServoSetX 3 11 4",
                "S ServoSetX 4 11 4",
                "S ServoSetX 1 65 1",
                "S ServoSetX 3 65 1",
                "S ServoSetX 4 65 1",
                "S ServoSetX 1 31 70",
                "S ServoSetX 3 31 70",
                "S ServoSetX 4 31 70",
                "S ServoSetX 1 63 52",
                "S ServoSetX 3 63 52",
                "S ServoSetX 4 63 52",
                "S ServoSetX 1 64 1",
                "S ServoSetX 3 64 1",
                "S ServoSetX 4 64 1",
                "S ServoSetX 4 84 50",
                "S ServoSetX 4 116 12 %54%08%00%00",
                "a 0 0 0 0 0 8040 1972",
                # "S AngularSpeedStartAndEnd 10000", 
                "S AngularSpeed 90000",
                # "S AngularAcceleration 10000",
                "S J1_PID_P 0.10",
                "S J2_PID_P 0.10",
                "S J3_PID_P 0.10",
                "S J4_PID_P 0.10",
                "S J5_PID_P 0.10"
                ]
        
        print("Initialization Started")
        
        for i in init_config_lst:
            command = f"xxx xxx xxx xxx {i};"
            self.send_command(command)
            time.sleep(0.5)
            
        robot_initialized = 1;
                        
        print("initialisation Finished")
       
    def __del__(self):
        # print("RevobotRobotBus.__del__ called")
        self.disconnect()

        
def main():
    robot_config = RevobotMotorsBusConfig(
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
    )
    robot = RobotInitialise(robot_config)
    robot.connect()
    print("initialisation 1/2")
    robot.write_init()
    print("3 sec rest")
    time.sleep(3)
    print("initialisation 2/2")
    robot.write_init()
    print("done")
    
    
        
if __name__ == "__main__":
    main()
    
    