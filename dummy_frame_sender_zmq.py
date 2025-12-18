#!/usr/bin/env python3
"""ZMQ video sender - streams frames from a video file."""

import os
import time
import cv2
import zmq

import sam3

# Configuration
ZMQ_ENDPOINT = "tcp://localhost:5555"
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
VIDEO_PATH = os.path.join(sam3_root, "assets", "videos", "bedroom.mp4")
TARGET_FPS = 30  # Set to 0 to send as fast as possible


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(ZMQ_ENDPOINT)
    print(f"Connected to {ZMQ_ENDPOINT}")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Streaming {VIDEO_PATH}: {width}x{height} @ {fps} FPS")
    
    frame_delay = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        sock.send(frame_rgb.tobytes())
        frame_idx += 1
        print(f"Sent frame {frame_idx}", end="\r")
        
        if frame_delay > 0:
            time.sleep(frame_delay)
    
    sock.send(b"END")
    cap.release()
    
    elapsed = time.time() - start_time
    print(f"\nSent {frame_idx} frames in {elapsed:.2f}s => {frame_idx/elapsed:.2f} FPS")
    
    time.sleep(0.5)  # Allow final message to flush
    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()