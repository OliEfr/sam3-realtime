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
import msgpack

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

    Expects msgpack-encoded messages with format:
        {
            "prompt": str,           # Text prompt for segmentation
            "is_first_frame": bool,  # True triggers session reset
            "frames": bytes          # Raw bytes: (batch_size * height * width * 3) uint8 RGB
        }

    Args:
        endpoint: ZMQ endpoint to bind to.
        batch_size: Number of frames per batch (e.g., number of cameras).
        height: Height of each frame.
        width: Width of each frame.

    Yields:
        dict with keys: "prompt", "is_first_frame", "frames" (np.ndarray of shape (batch_size, height, width, 3))
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

            # Deserialize msgpack message
            try:
                data = msgpack.unpackb(msg, raw=False)
            except msgpack.UnpackException as e:
                print(f"Warning: failed to unpack msgpack message: {e}. Skipping.")
                continue

            prompt = data.get("prompt", "")
            is_first_frame = data.get("is_first_frame", False)
            reset_model = data.get("reset_model", False)
            frame_bytes = data.get("frames", b"")

            # Handle reset_model message - create dummy frames to bypass validation
            if reset_model:
                dummy_frame_batch = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
                yield {
                    "prompt": "",
                    "is_first_frame": False,
                    "frames": dummy_frame_batch,
                    "reset_model": True
                }
                continue

            # Validate frame bytes
            if len(frame_bytes) != expected_bytes:
                print(f"Warning: received {len(frame_bytes)} frame bytes, expected {expected_bytes}. Skipping.")
                continue

            frame_batch = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((batch_size, height, width, 3))
            yield {
                "prompt": prompt,
                "is_first_frame": is_first_frame,
                "frames": frame_batch,
                "reset_model": reset_model
            }
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
    episode_idx = 0
    processed = 0
    peak_memory = 0.0
    start_time = time.time()
    current_prompt = ""

    # Publisher socket for segmented masks
    pub_ctx = zmq.Context()
    pub_sock = pub_ctx.socket(zmq.PUB)
    pub_sock.setsockopt(zmq.LINGER, 0)  # Allow immediate close without blocking
    pub_sock.bind(ZMQ_PUB_ENDPOINT)
    print(f"Publisher bound to {ZMQ_PUB_ENDPOINT}")

    async for msg in receive_frame_batches(ZMQ_RECV_ENDPOINT, BATCH_SIZE, HEIGHT, WIDTH):
        start_batch = time.time()

        # Extract message fields
        prompt = msg["prompt"]
        is_first_frame = msg["is_first_frame"]
        frame_batch = msg["frames"]
        reset_model = msg.get("reset_model", False)

        # Handle reset_model request
        if reset_model:
            for session_id in session_ids:
                predictor.handle_request({"type": "reset_session", "session_id": session_id})
            # Clear the publish socket queue by closing and recreating it
            pub_sock.setsockopt(zmq.LINGER, 0)  # Discard pending messages immediately
            pub_sock.close()
            time.sleep(0.1)  # Brief pause to ensure socket is closed
            pub_sock = pub_ctx.socket(zmq.PUB)
            pub_sock.setsockopt(zmq.LINGER, 0)
            pub_sock.bind(ZMQ_PUB_ENDPOINT)
            print("Received reset_model signal. All sessions reset and publish queue cleared")
            continue  # Skip to next message

        # Handle new episode (first frame of new sequence)
        if is_first_frame:
            # Only reset sessions if we've already processed at least one episode
            # (fresh sessions don't have state to reset)
            if episode_idx > 0:
                for session_id in session_ids:
                    predictor.handle_request({"type": "reset_session", "session_id": session_id})
            print(f"\nNew episode {episode_idx + 1} started, sessions reset. Prompt: '{prompt}'")
            episode_idx += 1
            frame_idx = 0
            current_prompt = prompt

        # Save input frames
        if SAVE_FRAMES:
            for i, cam_name in enumerate(CAMERA_NAMES):
                cv2.imwrite(
                    os.path.join(INPUT_FRAMES_DIR, cam_name, f"input_{episode_idx:03d}_{frame_idx:05d}.png"),
                    cv2.cvtColor(frame_batch[i], cv2.COLOR_RGB2BGR)
                )

        # Build batch requests
        batch_requests = [
            {"session_id": session_ids[i], "frame": frame_batch[i]}
            for i in range(BATCH_SIZE)
        ]

        if is_first_frame:
            # First frame of episode: add frames, then prompts, then run inference
            # (prompts must be added AFTER frames exist)
            predictor.batch_add_frame(frames=batch_requests)
            for session_id in session_ids:
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": current_prompt
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
                        os.path.join(OUTPUT_DIR, f"output_{CAMERA_NAMES[i]}_{episode_idx:03d}_{frame_idx:05d}.png"),
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

        print(f"Ep {episode_idx} Frame {frame_idx}, Total: {processed}, Peak mem: {current_peak:.2f} GB, FPS: {1/(end_batch - start_batch):.2f}")

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
    pub_sock.setsockopt(zmq.LINGER, 0)  # Allow immediate close without blocking
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