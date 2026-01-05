#!/usr/bin/env python3
"""SAM3 streaming predictor with async ZMQ frame receiving.

Supports both single-frame and batch-frame (multi-camera) modes.
"""

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
SAVE_FRAMES = True
FPS = 30
WIDTH, HEIGHT = 960, 540

CAMERA_NAMES = ["first_person", "third_person"]
BATCH_SIZE = len(CAMERA_NAMES)


async def receive_frame_batches(endpoint: str, batch_size: int, height: int, width: int):
    """Async generator that yields batches of frames from ZMQ.

    Args:
        endpoint: ZMQ endpoint to bind to.
        batch_size: Number of frames per batch (e.g., number of cameras).
        height: Height of each frame.
        width: Width of each frame.

    Yields:
        np.ndarray of shape (batch_size, height, width, 3)
    """
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
    sock.bind(endpoint)
    print(f"Receiver bound to {endpoint}, waiting for frame batches (batch_size={batch_size})...")

    expected_bytes = batch_size * height * width * 3

    try:
        while True:
            msg = await sock.recv()
            if msg == b"END":
                print("\nReceived END signal")
                break

            # Reshape to batch of frames
            if len(msg) != expected_bytes:
                print(f"Warning: received {len(msg)} bytes, expected {expected_bytes}. Skipping.")
                continue

            frame_batch = np.frombuffer(msg, dtype=np.uint8).reshape((batch_size, height, width, 3))
            yield frame_batch
    finally:
        sock.close()
        ctx.term()

# LEGACY single-camera mode
async def receive_frames(endpoint: str):
    """Async generator that yields single frames from ZMQ (legacy single-camera mode)."""
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


async def main_batch():
    """Main loop for batch (multi-camera) processing."""
    print(f"Starting SAM3 ZMQ stream receiver (BATCH MODE: {BATCH_SIZE} cameras)...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    if SAVE_FRAMES:
        for cam_name in CAMERA_NAMES:
            os.makedirs(os.path.join(INPUT_FRAMES_DIR, cam_name), exist_ok=True)

    # Initialize predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam3_stream_predictor(device=device)

    # Start a session for each camera
    session_ids = []
    for cam_name in CAMERA_NAMES:
        resp = predictor.handle_request({"type": "start_session", "session_id": cam_name})
        session_ids.append(resp["session_id"])
        print(f"Started session for camera '{cam_name}': {resp['session_id']}")

    frame_idx = 0
    processed = 0
    peak_memory = 0.0
    start_time = time.time()

    # Publisher socket for segmented masks
    pub_ctx = zmq.Context()
    pub_sock = pub_ctx.socket(zmq.PUB)
    pub_sock.bind(ZMQ_PUB_ENDPOINT)
    print(f"Publisher bound to {ZMQ_PUB_ENDPOINT}")

    async for frame_batch in receive_frame_batches(ZMQ_RECV_ENDPOINT, BATCH_SIZE, HEIGHT, WIDTH):
        start_batch = time.time()
        

        # Save input frames
        if SAVE_FRAMES:
            for i, cam_name in enumerate(CAMERA_NAMES):
                cv2.imwrite(
                    os.path.join(INPUT_FRAMES_DIR, cam_name, f"input_{frame_idx:05d}.png"),
                    cv2.cvtColor(frame_batch[i], cv2.COLOR_RGB2BGR)
                )

        # Build batch requests
        batch_requests = [
            {"session_id": session_ids[i], "frame": frame_batch[i]}
            for i in range(BATCH_SIZE)
        ]

        if frame_idx == 0:
            # First frame: add frames, then prompts, then run inference
            # (prompts must be added AFTER frames exist)
            predictor.batch_add_frame(frames=batch_requests)
            for session_id in session_ids:
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": "a mug"
                })
            batch_resp = predictor.batch_run_inference(
                sessions=[{"session_id": sid} for sid in session_ids]
            )
        else:
            # Subsequent frames: add + inference in one call
            batch_resp = predictor.batch_process(requests=batch_requests)

        # Process results for each camera
        # Collect first mask from each camera (client only uses first mask)
        first_masks = []
        results_list = batch_resp["results"]
        for i in range(len(results_list)):
            result = results_list[i]
            if result["error"]:
                print(f"Error for camera {CAMERA_NAMES[i]}: {result['error']}")
                # Add empty mask for this camera
                first_masks.append(np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
                continue

            outputs = result["outputs"]
            if outputs:
                binary_masks = outputs.get("out_binary_masks")  # (n_masks, H, W)
                if binary_masks is not None and len(binary_masks) > 0:
                    # Take only first mask
                    first_masks.append(binary_masks[0].astype(np.uint8))
                else:
                    # No masks detected - add empty mask
                    first_masks.append(np.zeros((HEIGHT, WIDTH), dtype=np.uint8))

                # Save output frame
                if SAVE_FRAMES:
                    overlay = render_masklet_frame(
                        frame_batch[i], outputs, frame_idx=result["frame_index"], alpha=0.5
                    )
                    cv2.imwrite(
                        os.path.join(OUTPUT_DIR, f"output_{CAMERA_NAMES[i]}_{frame_idx:05d}.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    )
            else:
                # No outputs - add empty mask
                first_masks.append(np.zeros((HEIGHT, WIDTH), dtype=np.uint8))

        # Publish masks: shape (num_cameras, 1, H, W) for client compatibility
        if first_masks:
            # Stack to (num_cameras, H, W) then add n_masks=1 dimension
            combined_masks = np.stack(first_masks, axis=0)[:, np.newaxis, :, :]
            pub_sock.send(combined_masks.tobytes())

        # Update stats
        frame_idx += 1
        processed += 1
        current_peak = torch.cuda.max_memory_allocated() / 1024**3
        peak_memory = max(peak_memory, current_peak)
        end_batch = time.time()

        print(f"Batch {frame_idx}, Frames: {processed}, Peak mem: {current_peak:.2f} GB, FPS (per batch): {1/(end_batch - start_batch):.2f}", end="\r")

    # Cleanup
    pub_sock.close()
    pub_ctx.term()
    for session_id in session_ids:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    elapsed = time.time() - start_time
    print(f"\nProcessed {processed} frames ({frame_idx} batches) in {elapsed:.2f}s => {processed/elapsed:.2f} FPS (per batch)")

# LEGACY single-camera mode
async def main_single():
    """Main loop for single-camera processing (legacy mode)."""
    print("Starting SAM3 ZMQ stream receiver (SINGLE MODE)...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    if SAVE_FRAMES:
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

    # Publisher socket for segmented masks
    pub_ctx = zmq.Context()
    pub_sock = pub_ctx.socket(zmq.PUB)
    pub_sock.bind(ZMQ_PUB_ENDPOINT)
    print(f"Publisher bound to {ZMQ_PUB_ENDPOINT}")

    async for frame_rgb in receive_frames(ZMQ_RECV_ENDPOINT):
        # Save input frame
        if SAVE_FRAMES:
            cv2.imwrite(
                os.path.join(INPUT_FRAMES_DIR, f"input_{frame_idx:05d}.png"),
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            )

        # Push frame
        predictor.handle_request({"type": "add_frame", "session_id": session_id, "frame": frame_rgb})

        # Add text prompt on first frame
        if frame_idx == 0:
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": "plate"
            })

        # Run inference
        resp = predictor.handle_request({
            "type": "run_inference",
            "session_id": session_id,
            "frame_index": frame_idx
        })
        outputs = resp.get("outputs")
        binary_masks = outputs.get("out_binary_masks")  # (n_masks, H, W)
        overlay = render_masklet_frame(frame_rgb, outputs, frame_idx=frame_idx, alpha=0.5) if outputs else frame_rgb

        # Publish segmented masks
        pub_sock.send(binary_masks.tobytes())

        # Save output frame
        if SAVE_FRAMES:
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"output_{frame_idx:05d}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )

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
    import argparse
    parser = argparse.ArgumentParser(description="SAM3 ZMQ Stream Receiver")
    args = parser.parse_args()

    asyncio.run(main_batch())
    # LEGACY single-camera mode
    # asyncio.run(main_single())