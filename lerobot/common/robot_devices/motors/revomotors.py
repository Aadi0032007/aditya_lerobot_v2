#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 8 13:20:28 2025

@author: aadi
"""

import time
import socket
import struct
from typing import List, Optional, Tuple
from collections import namedtuple
import numpy as np



from lerobot.common.robot_devices.motors.configs import RevobotMotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

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


class RevobotMotorsBus:
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


    def find_motor_indices(self):
        # print("RevobotRobotBus.find_motor_indices called")
        return list(self.motors.keys())

    @property
    def motor_names(self):
        # print("RevobotRobotBus.motor_names property accessed")
        return list(self.motors.keys())

    @property
    def motor_models(self):
        # print("RevobotRobotBus.motor_models property accessed")
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self):
        # print("RevobotRobotBus.motor_indices property accessed")
        return [idx for idx, _ in self.motors.values()]
   
    
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
                recv_data = self.sock.recv(240)
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

    def send_custom_command(self, command: str) -> bool:
        n = self.send_command(command)
        if n < 0:
            return False

        pos = self.read()
        if pos is None:
            print("Problem....Cannot Get JOINT Position....Problem")
            return False

        return True


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
    
    def revobot_robot_offset(self, index: int, value: float) -> int:
        """
        Compute the offset for a given motor index and value.
        Different joints use different scaling and offset adjustments.
        """
        
        if index == 1:
            return int((90 - int(value)) * 3600)
        elif index == 3:
            return int((int(value) - 90) * 3600)
        elif index == 5:  # 6th position (0-based index)
            # return int((67.5 - int(value)) * 88.8889) #Super Gripper
            return int((116 - int(value)) * 71.1111)
        elif index == 6:  # 7th position (0-based index)
            # return int(value * 700)  #Super Gripper
            return int(1350 + value * 17.778)
        else:
            return int(value * 3600)

    def write(self, data_name:str, values=[], motor_names=None):
        global write_call_counter, pause_gripper_angle
        write_call_counter += 1
        
        values_list = np.array(values).tolist()
        # print(values_list)
        if len(values_list) < 7:
            values_list.insert(4, 0)
        
        command_parts = ["xxx xxx xxx xxx P"]
        for i, value in enumerate(values_list):
            computed = self.revobot_robot_offset(i, value)
            
            # Skip the programming the Gripper Motor1 as we will use FPGA command to Exevute it
            if i < 6:
                command_parts.append(str(computed))
                
            # This is where we get the offset value of the Gripper 1 Motor
            # We use the offset value of Gripper1 value to program the Gripper2 Value
            # Motor 6 is the Gripper-1 Motor
            
            if i == 6:
                
                # We need to freeze motors 1-5 during training because
                # during gripping, human jitters are transferred to the robot motors
                freeze_flag = 0
                if ((value < 30) & (value > -5)):
                    freeze_flag = 1
                    
                # This is Gripper-1 steps in little endian. Notice 
                # that we already have offset value for this gripper
                byte_data = (computed).to_bytes(2, 'little')
                data3 = format(byte_data[0], '02x')
                data4 = format(byte_data[1], '02x')
                        
                # This is Gripper-2 steps in little endian. We need to calculate the 
                # offset value of this gripper based on the Gripper-1 offset. This is 
                # because in _offset() function only Motor 1-6 gripper offsets are calculated.
                computed = 1954 + int((45-value) * 17.778)
                byte_data = (computed).to_bytes(2, 'little')
                data1 = format(byte_data[0], '02x')
                data2 = format(byte_data[1], '02x')             
                
                # Gripper-2 Command
                command2 = "xxx xxx xxx xxx S ServoSetX 4 116 12 %"+str(data1)+"%"+str(data2)+"%00%00;"
                # Gripper-1 Command
                command3 = "xxx xxx xxx xxx S ServoSetX 1 116 12 %"+str(data3)+"%"+str(data4)+"%00%00;"
                
        command = " ".join(command_parts) + ";"
        
        # print(command)
        if  write_call_counter == 1:
            self.send_command(command2)
            self.send_command(command3)
            if (freeze_flag == 0):
                self.send_command(command)
            write_call_counter = 0
        else:
            self.read()
        # time.sleep(0.25)
        
       
    def __del__(self):
        # print("RevobotRobotBus.__del__ called")
        self.disconnect()
