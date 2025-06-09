# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 09:46:00 2025

@author: aadi
"""
import base64
import json
import threading
import time
from pathlib import Path

import cv2
import zmq


def setup_zmq_sockets(config):
    context = zmq.Context()

    video_socket = context.socket(zmq.PUSH)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.bind(f"tcp://*:{config.video_port}")

    return context, video_socket


def run_camera_capture(cameras, images_lock, latest_images_dict, stop_event):
    while not stop_event.is_set():
        local_dict = {}
        for name, cam in cameras.items():
            frame = cam.async_read()
            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ret:
                local_dict[name] = base64.b64encode(buffer).decode("utf-8")
            else:
                local_dict[name] = ""
        with images_lock:
            latest_images_dict.update(local_dict)
        time.sleep(0.01)



def run_revobot(robot_config):
    """
    Runs the Revobot robot:
      - Sets up cameras and connects them.
      - Creates ZeroMQ sockets for receiving commands and streaming observations.
    """
    # Import helper functions and classes
    from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs

    # Initialize cameras from the robot configuration.
    cameras = make_cameras_from_configs(robot_config.cameras)
    for cam in cameras.values():
        cam.connect()


    # Set up ZeroMQ sockets.
    context, cmd_socket, video_socket = setup_zmq_sockets(robot_config)

    # Start the camera capture thread.
    latest_images_dict = {}
    images_lock = threading.Lock()
    stop_event = threading.Event()
    cam_thread = threading.Thread(
        target=run_camera_capture, args=(cameras, images_lock, latest_images_dict, stop_event), daemon=True
    )
    cam_thread.start()

    last_cmd_time = time.time()
    print("Revobot Camera server started. Waiting for commands...")

    try:
        while True:
            loop_start_time = time.time()

            # Process incoming commands (non-blocking).
            while True:
                try:
                    msg = cmd_socket.recv_string(zmq.NOBLOCK)
                except zmq.Again:
                    break
            

            # Watchdog: stop the robot if no command is received for over 0.5 seconds.
            now = time.time()
            if now - last_cmd_time > 0.5:
                last_cmd_time = now

            # Get the latest camera images.
            with images_lock:
                images_dict_copy = dict(latest_images_dict)

            # Build the observation dictionary.
            observation = {
                "images": images_dict_copy
            }
            # Send the observation over the video socket.
            video_socket.send_string(json.dumps(observation))

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time
            # time.sleep(
            #     max(0.033 - elapsed, 0)
            # )  # If robot jitters increase the sleep and monitor cpu load with `top` in cmd
    except KeyboardInterrupt:
        print("Shutting down Revobot server.")
    finally:
        stop_event.set()
        cam_thread.join()
        cmd_socket.close()
        video_socket.close()
        context.term()