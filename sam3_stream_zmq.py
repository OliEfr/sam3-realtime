#!/usr/bin/env python3
"""SAM3 streaming predictor with async ZMQ frame receiving."""

import os
import shutil
import asyncio
import cv2
import numpy as np
import torch
import zmq
import zmq.asyncio
import time

import sam3
from sam3.model_builder import build_sam3_stream_predictor
from sam3.visualization_utils import render_masklet_frame

# Configuration
ZMQ_RECV_ENDPOINT = "tcp://*:5555"
ZMQ_PUB_ENDPOINT = "tcp://*:5556"
OUTPUT_DIR = "outputs/real_time/zmq_stream"
INPUT_FRAMES_DIR = os.path.join(OUTPUT_DIR, "input_frames")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "zmq_stream.mp4")
SAVE_INPUT_FRAMES = True
SAVE_OUTPUT_FRAMES = True
FPS = 30
WIDTH, HEIGHT = 960, 540


async def receive_frames(endpoint: str):
    """Async generator that yields frames from ZMQ."""
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
    sock.bind(endpoint)
    print(f"Receiver bound to {endpoint}, waiting for frames...")
    
    try:
        while True:
            msg = await sock.recv()
            if msg == b"END":
                print("\nReceived END signal")
                break
            frame = np.frombuffer(msg, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            yield frame
    finally:
        sock.close()
        ctx.term()


async def main():
    print("Starting SAM3 ZMQ stream receiver...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    if SAVE_INPUT_FRAMES:
        os.makedirs(INPUT_FRAMES_DIR)
    
    # Initialize predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam3_stream_predictor(device=device)
    resp = predictor.handle_request({"type": "start_session"})
    session_id = resp["session_id"]
    
    # Video writer
    writer = None
    if OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))
    
    frame_idx = 0
    processed = 0
    peak_memory = 0.0
    start_time = time.time()
    
    # Publisher socket for segmented masks (sync context - PUB send is non-blocking)
    pub_ctx = zmq.Context()
    pub_sock = pub_ctx.socket(zmq.PUB)
    pub_sock.bind(ZMQ_PUB_ENDPOINT)
    print(f"Publisher bound to {ZMQ_PUB_ENDPOINT}")

    async for frame_rgb in receive_frames(ZMQ_RECV_ENDPOINT):
        # Save input frame
        if SAVE_INPUT_FRAMES:
            cv2.imwrite(os.path.join(INPUT_FRAMES_DIR, f"input_{frame_idx:05d}.png"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        # Push frame
        predictor.handle_request({"type": "add_frame", "session_id": session_id, "frame": frame_rgb})
        
        # Add text prompt on first frame
        if frame_idx == 0:
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": "white and yellow cup"
            })
        
        # Run inference
        resp = predictor.handle_request({
            "type": "run_inference",
            "session_id": session_id,
            "frame_index": frame_idx
        })
        outputs = resp.get("outputs")
        binary_masks = outputs.get("out_binary_masks") # (n_masks, H, W)
        overlay = render_masklet_frame(frame_rgb, outputs, frame_idx=frame_idx, alpha=0.5) if outputs else frame_rgb
        
        # Publish segmented overlay
        pub_sock.send(binary_masks.tobytes())

        # Save output frame
        if SAVE_OUTPUT_FRAMES:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"output_{frame_idx:05d}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Write to video
        if writer:
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Update stats
        frame_idx += 1
        processed += 1
        current_peak = torch.cuda.max_memory_allocated() / 1024**3
        peak_memory = max(peak_memory, current_peak)
        print(f"Frame {frame_idx}, Peak mem: {current_peak:.2f} GB", end="\r")
    
    # Cleanup
    pub_sock.close()
    pub_ctx.term()
    if writer:
        writer.release()
    predictor.handle_request({"type": "close_session", "session_id": session_id})
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {processed} frames in {elapsed:.2f}s => {processed/elapsed:.2f} FPS")


if __name__ == "__main__":
    asyncio.run(main())