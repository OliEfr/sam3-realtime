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
import threading
import queue

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
WIDTH, HEIGHT = 256, 256

CAMERA_NAMES = ["1st_person", "3rd_person"]
# CAMERA_NAMES = [ "3rd_person"]
BATCH_SIZE = len(CAMERA_NAMES)

# Visualization configuration
ENABLE_VISUALIZATION = True  # Set to True to enable real-time display
VIZ_QUEUE_SIZE = 5  # Small queue - drop old frames if viewer is slow
VIZ_WINDOW_NAME = "SAM3 Third-Person View"


async def receive_frame_batches(endpoint: str, batch_size: int, height: int, width: int):
    """Async generator that yields batches of frames from ZMQ.

    Expects msgpack-encoded messages with format:
        {
            "prompt": str or dict,   # dict: {camera_name: prompt} for per-camera prompts
            "reset_model": bool,     # True triggers episode reset
            "frames": bytes          # Raw bytes: (batch_size * height * width * 3) uint8 RGB
        }

    Note: Prompt changes are detected automatically by comparing with previous prompts.

    Args:
        endpoint: ZMQ endpoint to bind to.
        batch_size: Number of frames per batch (e.g., number of cameras).
        height: Height of each frame.
        width: Width of each frame.

    Yields:
        dict with keys: "prompt", "reset_model", "frames" (np.ndarray of shape (batch_size, height, width, 3))
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

            prompt_field = data.get("prompt", "")
            reset_model = data.get("reset_model", False)
            frame_bytes = data.get("frames", b"")

            prompts = {}
            print(prompt_field)
            if prompt_field == "" or prompt_field is None or not prompt_field or prompt_field == {}:
                prompts = {camera: "" for camera in CAMERA_NAMES}
            elif isinstance(prompt_field, dict):
                for camera in CAMERA_NAMES:
                    if camera in prompt_field:
                        prompts[camera] = prompt_field[camera]
                    else:
                        raise ValueError(f"Missing prompt for camera '{camera}' in prompt dict.")
            else:
                raise ValueError(f"Invalid prompt field: {prompt_field}")

            # Handle reset_model message - create dummy frames to bypass validation
            if reset_model:
                dummy_frame_batch = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
                yield {
                    "prompt": {camera: "" for camera in CAMERA_NAMES},
                    "frames": dummy_frame_batch,
                    "reset_model": True
                }
                continue

            # Validate frame bytes
            assert len(frame_bytes) == expected_bytes, f"Warning: received {len(frame_bytes)} frame bytes, expected {expected_bytes}. Skipping."

            frame_batch = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((batch_size, height, width, 3))
            yield {
                "prompt": prompts,  # dict mapping camera names to prompts
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


class VisualizationThread:
    """Non-blocking visualization thread for displaying overlay frames.

    Runs independently from the main async loop to avoid blocking SAM3 processing.
    """

    def __init__(self, window_name: str, queue_size: int = 5):
        """Initialize the visualization thread.

        Args:
            window_name: Name of the cv2 window.
            queue_size: Max frames to buffer (older frames dropped on overflow).
        """
        self.window_name = window_name
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.thread = None
        self.running = False
        self.stop_event = threading.Event()

    def start(self):
        """Start the visualization thread."""
        if self.thread is not None:
            return  # Already running

        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"Visualization thread started (window: '{self.window_name}')")

    def _run(self):
        """Thread worker - displays frames from queue."""
        while not self.stop_event.is_set():
            try:
                # Wait for frame with timeout to allow checking stop_event
                frame_bgr, frame_info = self.frame_queue.get(timeout=0.1)

                # Display the frame
                try:
                    cv2.imshow(self.window_name, frame_bgr)
                except cv2.error as e:
                    print(f"cv2.imshow failed (no display?): {e}")
                    print("Stopping visualization - consider running without --viz flag")
                    break

                # waitKey(1) processes GUI events and returns key code
                # This MUST be called in the same thread as imshow
                key = cv2.waitKey(1) & 0xFF

                # Optional: Allow 'q' to request shutdown
                if key == ord('q'):
                    print("\nVisualization window closed by user (pressed 'q')")
                    break

            except queue.Empty:
                # No frame available, continue waiting
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nVisualization window closed by user (pressed 'q')")
                    break
                continue
                continue
            except Exception as e:
                print(f"Visualization thread error: {e}")
                break

        # Cleanup
        cv2.destroyWindow(self.window_name)
        self.running = False
        print("Visualization thread stopped")

    def put_frame(self, frame_bgr: np.ndarray, frame_info: dict = {}):
        """Put a frame in the display queue (non-blocking).

        Args:
            frame_bgr: Frame in BGR format (cv2 convention).
            frame_info: Optional metadata (episode_idx, frame_idx, etc.).

        Returns:
            bool: True if frame was added, False if queue is full (frame dropped).
        """
        if not self.running:
            return False

        try:
            # Put without blocking - raises queue.Full if queue is full
            self.frame_queue.put_nowait((frame_bgr, frame_info))
            return True
        except queue.Full:
            # Queue is full - drop this frame (acceptable per requirements)
            return False

    def stop(self):
        """Stop the visualization thread gracefully."""
        if not self.running:
            return

        print("Stopping visualization thread...")
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)  # Wait up to 2s for thread to finish
            if self.thread.is_alive():
                print("Warning: Visualization thread did not stop cleanly")
            self.thread = None


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
    current_prompts = {camera: "" for camera in CAMERA_NAMES}

    # Publisher socket for segmented masks
    pub_ctx = zmq.Context()
    pub_sock = pub_ctx.socket(zmq.PUB)
    pub_sock.setsockopt(zmq.LINGER, 0)  # Allow immediate close without blocking
    pub_sock.bind(ZMQ_PUB_ENDPOINT)
    print(f"Publisher bound to {ZMQ_PUB_ENDPOINT}")

    
    if ENABLE_VISUALIZATION:
        viz_thread = VisualizationThread(
            window_name=VIZ_WINDOW_NAME,
            queue_size=VIZ_QUEUE_SIZE
        )
        viz_thread.start()

    async for msg in receive_frame_batches(ZMQ_RECV_ENDPOINT, BATCH_SIZE, HEIGHT, WIDTH):
        start_batch = time.time()

        # Extract message fields
        prompts = msg["prompt"]  # dict mapping camera names to prompts
        frame_batch = msg["frames"]
        reset_model = msg.get("reset_model", False)


        is_new_prompt = prompts != current_prompts

        # handle session resets for new prompt or new episode
        if reset_model or is_new_prompt:
            # Only reset sessions if we've processed at least one frame
            if current_prompts != {camera: "" for camera in CAMERA_NAMES}:
                for session_id in session_ids:
                    predictor.handle_request({"type": "reset_session", "session_id": session_id})

        # handle reset_model (new episode)
        if reset_model:
            # Clear the publish socket queue by closing and recreating it
            pub_sock.setsockopt(zmq.LINGER, 0)  # Discard pending messages immediately
            pub_sock.close()
            time.sleep(0.1)  # Brief pause to ensure socket is closed
            pub_sock = pub_ctx.socket(zmq.PUB)
            pub_sock.setsockopt(zmq.LINGER, 0)
            pub_sock.bind(ZMQ_PUB_ENDPOINT)
            episode_idx += 1
            frame_idx = 0
            print(f"[RESET EPISODE]: Episode {episode_idx}: reset_model received.")
            continue  # Skip to next message

        # Handle prompt change (triggers session reset)
        if is_new_prompt:
            print(f"[PROMPT CHANGE] Detected prompt change, resetting sessions. Old: {current_prompts}, New: {prompts}")
            current_prompts = prompts

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

        if is_new_prompt:
            # First frame after prompt change: add frames, then prompts, then run inference
            # (prompts must be added AFTER frames exist)
            predictor.batch_add_frame(frames=batch_requests)
            for i, session_id in enumerate(session_ids):
                camera_name = CAMERA_NAMES[i]
                predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": current_prompts[camera_name]  # Different per camera
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

                # Save and/or display output frame for 3rd_person camera
                overlay = render_masklet_frame(
                    frame_batch[i], outputs, frame_idx=result["frame_index"], alpha=0.5
                )

                if SAVE_FRAMES:
                    cv2.imwrite(
                        os.path.join(OUTPUT_DIR, f"output_{CAMERA_NAMES[i]}_{episode_idx:03d}_{frame_idx:05d}.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    )

                # Display in window (non-blocking)
                if CAMERA_NAMES[i] == "3rd_person" and ENABLE_VISUALIZATION:
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    viz_thread.put_frame(overlay_bgr, {
                        "episode": episode_idx,
                        "frame": frame_idx,
                        "camera": CAMERA_NAMES[i]
                    })
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

        print(f"Ep {episode_idx} Frame {frame_idx}, Total: {processed}, Peak mem: {current_peak:.2f} GB, FPS: {1/(end_batch - start_batch):.2f}, Prompts: {current_prompts}")

    # Cleanup
    pub_sock.close()
    pub_ctx.term()
    for session_id in session_ids:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    # Cleanup visualization thread
    if viz_thread is not None:
        viz_thread.stop()

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
   

    asyncio.run(main_batch())
    # LEGACY single-camera mode
    # asyncio.run(main_single())