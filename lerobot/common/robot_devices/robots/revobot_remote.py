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
    cmd_socket = context.socket(zmq.PULL)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.bind(f"tcp://*:{config.port}")

    video_socket = context.socket(zmq.PUSH)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.bind(f"tcp://*:{config.video_port}")

    return context, cmd_socket, video_socket


def run_camera_capture(cameras, images_lock, latest_images_dict, stop_event):
    while not stop_event.is_set():
        local_dict = {}
        cam_start_all = time.time()  # --- TIMING ---
        for name, cam in cameras.items():
            cam_start = time.time()  # --- TIMING ---
            frame = cam.async_read()
            cam_elapsed = time.time() - cam_start  # --- TIMING ---
            print(f"[CAMERA] Time to get 1 frame from '{name}': {cam_elapsed:.4f}s")  # --- TIMING ---

            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ret:
                local_dict[name] = base64.b64encode(buffer).decode("utf-8")
            else:
                local_dict[name] = ""
        cam_all_elapsed = time.time() - cam_start_all  # --- TIMING ---
        print(f"[CAMERA] Time to get all camera frames: {cam_all_elapsed:.4f}s")  # --- TIMING ---

        with images_lock:
            latest_images_dict.update(local_dict)
        time.sleep(0.005)


def run_revobot(robot_config):
    """
    Runs the Revobot robot:
      - Sets up cameras and connects them.
      - Creates ZeroMQ sockets for receiving commands and streaming observations.
    """
    from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs

    cameras = make_cameras_from_configs(robot_config.cameras)
    for cam in cameras.values():
        cam.connect()

    context, cmd_socket, video_socket = setup_zmq_sockets(robot_config)

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
            loop_start_time = time.time()  # --- TIMING ---

            # Time to receive command
            recv_start = time.time()  # --- TIMING ---
            while True:
                try:
                    msg = cmd_socket.recv_string(zmq.NOBLOCK)
                    print(f"[CMD] Received command: {msg}")  # --- TIMING ---
                except zmq.Again:
                    break
            recv_elapsed = time.time() - recv_start  # --- TIMING ---
            print(f"[ZMQ] Time to receive command: {recv_elapsed:.4f}s")  # --- TIMING ---

            now = time.time()
            if now - last_cmd_time > 0.5:
                last_cmd_time = now

            # Time to access shared image data
            image_lock_start = time.time()  # --- TIMING ---
            with images_lock:
                images_dict_copy = dict(latest_images_dict)
            image_lock_elapsed = time.time() - image_lock_start  # --- TIMING ---
            print(f"[LOCK] Time to copy image dict: {image_lock_elapsed:.4f}s")  # --- TIMING ---

            # Build and send observation
            obs = {"images": images_dict_copy}
            send_start = time.time()  # --- TIMING ---
            video_socket.send_string(json.dumps(obs))
            send_elapsed = time.time() - send_start  # --- TIMING ---
            print(f"[ZMQ] Time to send video data: {send_elapsed:.4f}s")  # --- TIMING ---

            # Total loop time
            loop_elapsed = time.time() - loop_start_time  # --- TIMING ---
            print(f"[LOOP] Total time for loop iteration: {loop_elapsed:.4f}s\n")  # --- TIMING ---
            print("")
            print("")
            print("")
            print("")
            print("")
            print("")
            
            print(max(0.033 - loop_elapsed, 0))
            time.sleep(max(0.033 - loop_elapsed, 0))

    except KeyboardInterrupt:
        print("Shutting down Revobot server.")
    finally:
        stop_event.set()
        cam_thread.join()
        cmd_socket.close()
        video_socket.close()
        context.term()
