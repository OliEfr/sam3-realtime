# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import gc
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import torch

from sam3.logger import get_logger


logger = get_logger(__name__)


class Sam3StreamPredictor:
    """Single-GPU streaming predictor that mirrors Sam3VideoPredictor API.

    This wraps the real-time Sam3StreamInference, providing session management
    and a request-based interface tailored for frame-by-frame streaming.

    Exposed request types:
      Single-session:
      - {"type": "start_session", "session_id": Optional[str]}
      - {"type": "add_frame", "session_id": str, "frame": raw_image}
      - {"type": "add_prompt", "session_id": str, "frame_index": int, "text": Optional[str],
         "bounding_boxes": Optional[List[List[float]]], "bounding_box_labels": Optional[List[int]]}
      - {"type": "run_inference", "session_id": str, "frame_index": Optional[int]}
      - {"type": "get_cached_output", "session_id": str, "frame_index": int}
      - {"type": "reset_session", "session_id": str}
      - {"type": "close_session", "session_id": str}
      - {"type": "warm_up_compilation"}

      Multi-session batch (parallel processing):
      - {"type": "batch_add_frame", "frames": [{"session_id": str, "frame": raw_image}, ...]}
      - {"type": "batch_run_inference", "sessions": [{"session_id": str, "frame_index": Optional[int]}, ...]}
      - {"type": "batch_process", "requests": [{"session_id": str, "frame": raw_image}, ...]}
    """

    _ALL_INFERENCE_STATES = {}
    _STATES_LOCK = threading.RLock()  # Protects access to _ALL_INFERENCE_STATES

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
        has_presence_token: bool = True,
        geo_encoder_use_img_cross_attn: bool = True,  # kept for parity; not used directly here
        strict_state_dict_loading: bool = True,
        apply_temporal_disambiguation: bool = True,
        device: Optional[str] = None,
        compile: bool = False,
        max_sessions: int = 8,
        preprocessing_workers: int = 4,
    ) -> None:

        from sam3.model_builder import build_sam3_stream_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = (
            build_sam3_stream_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=True if checkpoint_path is None else False,
                bpe_path=bpe_path,
                has_presence_token=has_presence_token,
                geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
                strict_state_dict_loading=strict_state_dict_loading,
                apply_temporal_disambiguation=apply_temporal_disambiguation,
                device=device,
                compile=compile,
            )
            .to(device=device)
            .eval()
        )

        # Multi-session batch processing configuration
        self._max_sessions = max_sessions
        self._session_locks: Dict[str, threading.RLock] = {}
        self._preprocessing_executor = ThreadPoolExecutor(max_workers=preprocessing_workers)

    @torch.inference_mode()
    def handle_request(self, request):
        request_type = request["type"]
        if request_type == "start_session":
            return self.start_session(session_id=request.get("session_id"))
        elif request_type == "add_frame":
            return self.add_frame(
                session_id=request["session_id"],
                frame=request["frame"],
            )
        elif request_type == "add_prompt":
            return self.add_prompt(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
                text=request.get("text"),
                bounding_boxes=request.get("bounding_boxes"),
                bounding_box_labels=request.get("bounding_box_labels"),
            )
        elif request_type == "run_inference":
            return self.run_inference(
                session_id=request["session_id"],
                frame_idx=request.get("frame_index"),
            )
        elif request_type == "get_cached_output":
            return self.get_cached_output(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
            )
        elif request_type == "reset_session":
            return self.reset_session(session_id=request["session_id"])
        elif request_type == "close_session":
            return self.close_session(session_id=request["session_id"])
        elif request_type == "warm_up_compilation":
            return self.warm_up_compilation()
        # Batch request types for multi-session parallel processing
        elif request_type == "batch_add_frame":
            return self.batch_add_frame(frames=request["frames"])
        elif request_type == "batch_run_inference":
            return self.batch_run_inference(sessions=request["sessions"])
        elif request_type == "batch_process":
            return self.batch_process(requests=request["requests"])
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    def start_session(self, session_id: Optional[str] = None):
        inference_state = self.model.init_stream_state()
        if not session_id:
            session_id = str(uuid.uuid4())

        with self._STATES_LOCK:
            if len(self._ALL_INFERENCE_STATES) >= self._max_sessions:
                raise RuntimeError(
                    f"Maximum number of sessions ({self._max_sessions}) reached. "
                    f"Close existing sessions before starting new ones."
                )
            self._ALL_INFERENCE_STATES[session_id] = {
                "state": inference_state,
                "session_id": session_id,
                "start_time": time.time(),
            }
            self._session_locks[session_id] = threading.RLock()

        logger.debug(
            f"started new stream session {session_id}; {self._get_session_stats()}; "
            f"{self._get_torch_and_gpu_properties()}"
        )
        return {"session_id": session_id}

    def add_frame(self, session_id: str, frame):
        session = self._get_session(session_id)
        inference_state = session["state"]
        frame_idx = self.model.add_frame(inference_state=inference_state, raw_image=frame)
        logger.debug(f"added frame -> session={session_id}, frame_index={frame_idx}")
        return {"frame_index": frame_idx}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: Optional[str] = None,
        bounding_boxes: Optional[list] = None,
        bounding_box_labels: Optional[list] = None,
    ):
        session = self._get_session(session_id)
        inference_state = session["state"]

        logger.debug(
            f"add prompt on frame {frame_idx} in session {session_id}: "
            f"text={text}, boxes={bounding_boxes}, box_labels={bounding_box_labels}"
        )
        frame_idx, outputs = self.model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text,
            boxes_xywh=bounding_boxes,
            box_labels=bounding_box_labels,
        )
        return {"frame_index": frame_idx, "outputs": outputs}

    def run_inference(self, session_id: str, frame_idx: Optional[int] = None):
        session = self._get_session(session_id)
        inference_state = session["state"]
        outputs = self.model.run_single_frame_inference(
            inference_state=inference_state, frame_idx=frame_idx
        )
        return {"frame_index": inference_state["curr_frame_idx"] if frame_idx is None else frame_idx, "outputs": outputs}

    def get_cached_output(self, session_id: str, frame_idx: int):
        session = self._get_session(session_id)
        inference_state = session["state"]
        cached = self.model.get_cached_output_for_frame(inference_state, frame_idx)
        return {"frame_index": frame_idx, "cached": cached}

    def reset_session(self, session_id: str):
        logger.debug(f"reset stream session {session_id}")
        session = self._get_session(session_id)
        self.model.reset_stream(session["state"])
        return {"is_success": True}

    def close_session(self, session_id: str):
        with self._STATES_LOCK:
            session = self._ALL_INFERENCE_STATES.pop(session_id, None)
            self._session_locks.pop(session_id, None)

        if session is None:
            logger.warning(
                f"cannot close session {session_id} as it does not exist (it might have expired); "
                f"{self._get_session_stats()}"
            )
        else:
            del session
            gc.collect()
            logger.info(f"removed session {session_id}; {self._get_session_stats()}")
        return {"is_success": True}

    def warm_up_compilation(self, num_frames: int = 8):
        self.model.warm_up_compilation(num_frames=num_frames)
        return {"is_success": True}

    def _get_session(self, session_id: str):
        with self._STATES_LOCK:
            session = self._ALL_INFERENCE_STATES.get(session_id, None)
        if session is None:
            raise RuntimeError(f"Cannot find session {session_id}; it might have expired")
        return session

    def _get_session_lock(self, session_id: str) -> threading.RLock:
        """Get the lock for a specific session (thread-safe)."""
        with self._STATES_LOCK:
            lock = self._session_locks.get(session_id)
            if lock is None:
                raise RuntimeError(f"Cannot find lock for session {session_id}; session might not exist")
            return lock

    def _get_session_stats(self):
        # print both the session ids and their frame numbers
        live_session_strs = [
            f"'{session_id}' ({session['state'].get('num_frames', 0)} frames)"
            for session_id, session in self._ALL_INFERENCE_STATES.items()
        ]
        session_stats_str = (
            f"live sessions: [{', '.join(live_session_strs)}], GPU memory: "
            f"{torch.cuda.memory_allocated() // 1024**2} MiB used and "
            f"{torch.cuda.memory_reserved() // 1024**2} MiB reserved"
            f" (max over time: {torch.cuda.max_memory_allocated() // 1024**2} MiB used "
            f"and {torch.cuda.max_memory_reserved() // 1024**2} MiB reserved)"
        )
        return session_stats_str

    def _get_torch_and_gpu_properties(self):
        torch_and_gpu_str = (
            f"torch: {torch.__version__} with CUDA arch {torch.cuda.get_arch_list()}, "
            f"GPU device: {torch.cuda.get_device_properties(torch.cuda.current_device())}"
        )
        return torch_and_gpu_str

    def shutdown(self):
        with self._STATES_LOCK:
            self._ALL_INFERENCE_STATES.clear()
            self._session_locks.clear()
        self._preprocessing_executor.shutdown(wait=False)

    # -------------------------------------------------------------------------
    # Multi-session batch processing methods
    # -------------------------------------------------------------------------

    def _batch_preprocess_frames(
        self, frames: List[Tuple[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Preprocess multiple frames from different sessions in parallel.

        Args:
            frames: List of (session_id, raw_image) tuples.

        Returns:
            Dict mapping session_id to {"tensor": img_t, "orig_height": h, "orig_width": w}
        """
        def preprocess_one(session_id: str, raw_image: Any):
            img_t, orig_h, orig_w = self.model._preprocess_raw_image(raw_image)
            return session_id, {"tensor": img_t, "orig_height": orig_h, "orig_width": orig_w}

        results = {}
        futures = [
            self._preprocessing_executor.submit(preprocess_one, sid, frame)
            for sid, frame in frames
        ]
        for future in as_completed(futures):
            session_id, data = future.result()
            results[session_id] = data
        return results

    @torch.inference_mode()
    def batch_add_frame(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add frames to multiple sessions in parallel.

        Preprocessing is parallelized across sessions; frame addition is sequential
        but fast since images are already preprocessed.

        Args:
            frames: List of {"session_id": str, "frame": raw_image}

        Returns:
            {"results": [{"session_id": str, "frame_index": int, "error": Optional[str]}, ...]}
        """
        # Preprocess all frames in parallel
        session_frames = [(f["session_id"], f["frame"]) for f in frames]
        preprocessed = self._batch_preprocess_frames(session_frames)

        results = []
        for item in frames:
            session_id = item["session_id"]
            try:
                session_lock = self._get_session_lock(session_id)
                with session_lock:
                    session = self._get_session(session_id)
                    inference_state = session["state"]
                    preproc = preprocessed[session_id]

                    # Add preprocessed frame to session
                    frame_idx = self._add_preprocessed_frame(
                        inference_state,
                        preproc["tensor"],
                        preproc["orig_height"],
                        preproc["orig_width"],
                    )
                    results.append({
                        "session_id": session_id,
                        "frame_index": frame_idx,
                        "error": None,
                    })
            except Exception as e:
                logger.error(f"batch_add_frame error for session {session_id}: {e}")
                results.append({
                    "session_id": session_id,
                    "frame_index": None,
                    "error": str(e),
                })

        logger.debug(f"batch_add_frame processed {len(frames)} frames across {len(set(f['session_id'] for f in frames))} sessions")
        return {"results": results}

    @torch.inference_mode()
    def batch_run_inference(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run inference on multiple sessions.

        Note: Inference runs sequentially per session on the GPU due to independent
        tracking state dependencies. Preprocessing and result collection are parallelized.

        Args:
            sessions: List of {"session_id": str, "frame_index": Optional[int]}

        Returns:
            {"results": [{"session_id": str, "frame_index": int, "outputs": dict, "error": Optional[str]}, ...]}
        """
        results = []
        for item in sessions:
            session_id = item["session_id"]
            frame_idx = item.get("frame_index")
            try:
                session_lock = self._get_session_lock(session_id)
                with session_lock:
                    session = self._get_session(session_id)
                    inference_state = session["state"]

                    outputs = self.model.run_single_frame_inference(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                    )
                    actual_frame_idx = (
                        frame_idx if frame_idx is not None
                        else inference_state["curr_frame_idx"]
                    )
                    results.append({
                        "session_id": session_id,
                        "frame_index": actual_frame_idx,
                        "outputs": outputs,
                        "error": None,
                    })
            except Exception as e:
                logger.error(f"batch_run_inference error for session {session_id}: {e}")
                results.append({
                    "session_id": session_id,
                    "frame_index": item.get("frame_index"),
                    "outputs": None,
                    "error": str(e),
                })

        logger.debug(f"batch_run_inference processed {len(sessions)} sessions")
        return {"results": results}

    @torch.inference_mode()
    def batch_process(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add frames and run inference for multiple sessions in one call.

        This is the primary method for multi-camera streaming scenarios where you
        want to process frames from multiple video streams simultaneously.

        Args:
            requests: List of {"session_id": str, "frame": raw_image}

        Returns:
            {"results": [{"session_id": str, "frame_index": int, "outputs": dict, "error": Optional[str]}, ...]}
        """
        # Step 1: Batch preprocess all frames in parallel
        session_frames = [(r["session_id"], r["frame"]) for r in requests]
        preprocessed = self._batch_preprocess_frames(session_frames)

        results = []
        # Step 2: Process each session (add frame + inference)
        for item in requests:
            session_id = item["session_id"]
            try:
                session_lock = self._get_session_lock(session_id)
                with session_lock:
                    session = self._get_session(session_id)
                    inference_state = session["state"]
                    preproc = preprocessed[session_id]

                    # Add preprocessed frame
                    frame_idx = self._add_preprocessed_frame(
                        inference_state,
                        preproc["tensor"],
                        preproc["orig_height"],
                        preproc["orig_width"],
                    )

                    # Run inference on this frame
                    outputs = self.model.run_single_frame_inference(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                    )

                    results.append({
                        "session_id": session_id,
                        "frame_index": frame_idx,
                        "outputs": outputs,
                        "error": None,
                    })
            except Exception as e:
                logger.error(f"batch_process error for session {session_id}: {e}")
                results.append({
                    "session_id": session_id,
                    "frame_index": None,
                    "outputs": None,
                    "error": str(e),
                })

        logger.debug(f"batch_process completed {len(requests)} requests")
        return {"results": results}

    def _add_preprocessed_frame(
        self,
        inference_state: Dict[str, Any],
        img_t: torch.Tensor,
        orig_h: int,
        orig_w: int,
    ) -> int:
        """Add an already-preprocessed frame tensor to inference state.

        This mirrors Sam3StreamInference.add_frame() but skips preprocessing
        since the image is already preprocessed.
        """
        from sam3.model.data_misc import BatchedDatapoint, FindStage, convert_my_tensors
        from sam3.model.utils.misc import copy_data_to_device

        if inference_state["orig_height"] is None:
            inference_state["orig_height"] = orig_h
            inference_state["orig_width"] = orig_w

        if inference_state["input_batch"] is None:
            # Initialize input batch (first frame)
            find_text_batch = ["<text placeholder>", "visual"]
            input_box_embedding_dim = 258
            input_points_embedding_dim = 257
            stage = FindStage(
                img_ids=[0],
                text_ids=[self.model.TEXT_ID_FOR_VISUAL],
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
                input_boxes_label=[torch.empty(0, dtype=torch.long)],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_mask=[torch.empty(0)],
                object_ids=[],
            )
            stage = convert_my_tensors(stage)
            img_batch = [img_t]
            input_batch = BatchedDatapoint(
                img_batch=img_batch,
                find_text_batch=find_text_batch,
                find_inputs=[copy_data_to_device(stage, self.model.device, non_blocking=True)],
                find_targets=[None],
                find_metadatas=[None],
            )
            inference_state["input_batch"] = input_batch
            inference_state["curr_frame_idx"] = 0
        else:
            # Append to existing batch
            input_batch = inference_state["input_batch"]
            if isinstance(input_batch.img_batch, torch.Tensor):
                T_prev = input_batch.img_batch.shape[0]
                input_batch.img_batch = torch.cat([input_batch.img_batch, img_t.unsqueeze(0)], dim=0)
            else:
                T_prev = len(input_batch.img_batch)
                input_batch.img_batch.append(img_t)

            input_box_embedding_dim = 258
            input_points_embedding_dim = 257
            text_id = (
                self.model.TEXT_ID_FOR_TEXT
                if inference_state.get("text_prompt")
                else self.model.TEXT_ID_FOR_VISUAL
            )
            stage = FindStage(
                img_ids=[T_prev],
                text_ids=[text_id],
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
                input_boxes_label=[torch.empty(0, dtype=torch.long)],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_mask=[torch.empty(0)],
                object_ids=[],
            )
            stage = convert_my_tensors(stage)
            stage = copy_data_to_device(stage, self.model.device, non_blocking=True)
            input_batch.find_inputs.append(stage)
            input_batch.find_targets.append(None)
            input_batch.find_metadatas.append(None)
            inference_state["curr_frame_idx"] = T_prev

        # Grow per-frame placeholders
        inference_state["previous_stages_out"].append(None)
        inference_state["per_frame_raw_point_input"].append(None)
        inference_state["per_frame_raw_box_input"].append(None)
        inference_state["per_frame_visual_prompt"].append(None)
        inference_state["per_frame_geometric_prompt"].append(None)
        inference_state["per_frame_cur_step"].append(0)

        # Update total frames and keep tracker states in sync
        img_batch_ref = inference_state["input_batch"].img_batch
        inference_state["num_frames"] = (
            img_batch_ref.shape[0] if isinstance(img_batch_ref, torch.Tensor) else len(img_batch_ref)
        )
        if inference_state["tracker_inference_states"]:
            for trk_state in inference_state["tracker_inference_states"]:
                trk_state["num_frames"] = inference_state["num_frames"]
                if trk_state.get("video_height", None) is None:
                    trk_state["video_height"] = inference_state["orig_height"]
                    trk_state["video_width"] = inference_state["orig_width"]

        return inference_state["curr_frame_idx"]
