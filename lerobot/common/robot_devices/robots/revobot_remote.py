# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 09:46:00 2025

@author: aadi
"""

import socket
import threading
import time
import json
import base64
import cv2


print_needed = False

def start_tcp_server(host: str, port: int) -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[TCP] Listening on {host}:{port}")
    cl, addr = srv.accept()
    print(f"[TCP] Client connected: {addr}")
    # disable Nagle
    cl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return cl


def encode_frame(frame) -> str:
    t0 = time.perf_counter()
    ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    elapsed = time.perf_counter() - t0
    if print_needed:
        print(f"[TIME] JPEG encode: {elapsed:.4f}s")
    if ret:
        return base64.b64encode(buffer).decode("utf-8")
    return ""


def run_camera_capture(cameras, images_lock, latest_images_dict, stop_event):
    """Capture frames into shared memory at high speed."""
    while not stop_event.is_set():
        t_loop = time.perf_counter()
        local_dict = {}
        for name, cam in cameras.items():
            t_cam = time.perf_counter()
            frame = cam.async_read()
            t_cam_elapsed = time.perf_counter() - t_cam
            if print_needed:
                print(f"[TIME] Camera read [{name}]: {t_cam_elapsed:.4f}s")

            encoded = encode_frame(frame)
            if encoded:
                local_dict[name] = encoded

        t_lock = time.perf_counter()
        with images_lock:
            latest_images_dict.clear()
            latest_images_dict.update(local_dict)
        t_lock_elapsed = time.perf_counter() - t_lock
        if print_needed:
            print(f"[TIME] Lock + update shared frame dict: {t_lock_elapsed:.4f}s")

        if print_needed:
            print(f"[CAMERA] Total camera loop: {time.perf_counter() - t_loop:.4f}s\n")
        time.sleep(0.005)  # fast loop


def run_video_stream_sender(client_socket, images_lock, latest_images_dict, stop_event):
    """Send latest frame every 33ms. Reconnect on failure."""
    try:
        while not stop_event.is_set():
            t_loop = time.perf_counter()

            with images_lock:
                if not latest_images_dict:
                    continue
                obs = {"images": dict(latest_images_dict)}

            t_json = time.perf_counter()
            payload = json.dumps(obs).encode("utf-8")
            if print_needed:
                print(f"[ENCODE] JSON encode: {time.perf_counter() - t_json:.4f}s")

            try:
                t_send = time.perf_counter()
                client_socket.sendall(len(payload).to_bytes(4, "big") + payload)
                if print_needed:
                    print(f"[SEND] Socket sendall: {time.perf_counter() - t_send:.4f}s")
                
                t_rcv_loop = time.perf_counter()
                ack = client_socket.recv(1)
                if ack != b'\x01':
                    break
                
                if print_needed:
                    print(f"[TIME] Total rcvr loop: {time.perf_counter() - t_rcv_loop:.4f}s\n")
            except (BrokenPipeError, ConnectionResetError, socket.error) as e:
                print(f"[TCP][DISCONNECT] Client lost: {e}")
                break  # Exit thread to allow reconnect

            if print_needed:
                print(f"[TIME] Total sender loop: {time.perf_counter() - t_loop:.4f}s\n")
            time.sleep(max(0.06 - (time.perf_counter() - t_loop), 0))

    finally:
        print("[TCP] Closing client socket.")
        client_socket.close()


def run_revobot(robot_config):
    """
    Runs camera + TCP server. Reconnects if client disconnects.
    Uses non-daemon threads and safe shutdown.
    """
    from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs

    cameras = make_cameras_from_configs(robot_config.cameras)
    for cam in cameras.values():
        cam.connect()

    latest_images_dict = {}
    images_lock = threading.Lock()
    stop_event = threading.Event()

    # Start camera capture thread once (persistent)
    print("[TCP] Starting camera thread...")
    cam_thread = threading.Thread(
        target=run_camera_capture,
        args=(cameras, images_lock, latest_images_dict, stop_event),
    )
    cam_thread.start()

    try:
        while not stop_event.is_set():
            print("[TCP] Waiting for new client...")
            client_socket = start_tcp_server("0.0.0.0", robot_config.video_port)

            print("[TCP] Starting video sender thread...")
            send_thread = threading.Thread(
                target=run_video_stream_sender,
                args=(client_socket, images_lock, latest_images_dict, stop_event),
            )
            send_thread.start()
            send_thread.join()  # Wait until disconnected

            print("[TCP] Client disconnected. Restarting listen...\n")

    except KeyboardInterrupt:
        print("[TCP] Server shutdown requested.")
    finally:
        stop_event.set()
        cam_thread.join()
        print("[TCP] Revobot server fully stopped.")
