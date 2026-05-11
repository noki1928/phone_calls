import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pyannote.audio import Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from torch.torch_version import TorchVersion

from .preprocess import load_audio

_PIPELINE = None
_PIPELINE_PARAMS = None


class Binarize:
    def __init__(
        self,
        onset: float = 0.5,
        offset: Optional[float] = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        max_duration: float = float("inf"),
    ):
        self.onset = onset
        self.offset = offset or onset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.max_duration = max_duration

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        num_frames, _ = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        active = Annotation()
        for label_idx, label_scores in enumerate(scores.data.T):
            label = label_idx if scores.labels is None else scores.labels[label_idx]
            start = timestamps[0]
            is_active = label_scores[0] > self.onset
            curr_scores = [label_scores[0]]
            curr_timestamps = [start]
            t = start

            for t, score in zip(timestamps[1:], label_scores[1:]):
                if is_active:
                    curr_duration = t - start
                    if curr_duration > self.max_duration:
                        search_after = len(curr_scores) // 2
                        split_idx = search_after + int(np.argmin(curr_scores[search_after:]))
                        split_time = curr_timestamps[split_idx]
                        active[Segment(start, split_time), label_idx] = label
                        start = split_time
                        curr_scores = curr_scores[split_idx + 1 :]
                        curr_timestamps = curr_timestamps[split_idx + 1 :]
                    elif score < self.offset:
                        active[Segment(start, t), label_idx] = label
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []

                    curr_scores.append(score)
                    curr_timestamps.append(t)
                elif score > self.onset:
                    start = t
                    is_active = True

            if is_active:
                active[Segment(start, t), label_idx] = label

        if self.min_duration_off > 0.0:
            active = active.support(collar=self.min_duration_off)

        if self.min_duration_on > 0.0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active


class VoiceActivitySegmentation(VoiceActivityDetection):
    def apply(self, file: AudioFile, hook=None) -> SlidingWindowFeature:
        hook = self.setup_hook(file, hook=hook)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                return file[self.CACHED_SEGMENTATION]
            segmentations = self._segmentation(file)
            file[self.CACHED_SEGMENTATION] = segmentations
            return segmentations
        return self._segmentation(file)


def resolve_local_segmentation_path(model_id: str) -> str:
    """
    Finds the local path to the segmentation model.
    """
    try:
        return snapshot_download(
            repo_id=model_id,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pass

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            f"Model {model_id} was not found locally, "
            f"and no HF_TOKEN was provided to download it."
        )

    return snapshot_download(
        repo_id=model_id,
        token=hf_token,
    )


def load_segmentation_model(model_id: str) -> Model:
    """
    Loads the segmentation model from a local snapshot.
    If it doesn’t exist, it first creates (downloads) the snapshot.
    """
    local_path = resolve_local_segmentation_path(model_id=model_id)

    with torch.serialization.safe_globals(
        [
            TorchVersion,
            Problem,
            Specifications,
            Resolution,
        ]
    ):
        return Model.from_pretrained(local_path)


def get_pipeline(
    device: torch.device,
    model_id: str = "pyannote/segmentation-3.0",
    vad_onset: float = 0.500,
    vad_offset: float = 0.363,
) -> Pipeline:
    """
    Retrieves a PyAnnote voice activity detection pipeline and moves it to the specified device.
    The pipeline is loaded only once and reused across subsequent calls.
    It requires the Hugging Face API token to be set in the HF_TOKEN environment variable.
    """
    global _PIPELINE, _PIPELINE_PARAMS
    pipeline_params = (model_id, vad_onset, vad_offset)
    if _PIPELINE is not None and _PIPELINE_PARAMS == pipeline_params:
        return _PIPELINE.to(device)

    model = load_segmentation_model(model_id=model_id)

    _PIPELINE = VoiceActivitySegmentation(segmentation=model)
    _PIPELINE.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
    _PIPELINE_PARAMS = pipeline_params

    return _PIPELINE.to(device)


def segment_audio_file(
    wav_file: str,
    sr: int,
    chunk_size: float = 30.0,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    strict_limit_duration: Optional[float] = None,
    new_chunk_threshold: float = 0.2,
    vad_onset: float = 0.500,
    vad_offset: float = 0.363,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """
    Segments an audio waveform into smaller chunks based on speech activity.
    The segmentation is performed using a PyAnnote voice activity detection pipeline.
    """

    max_duration = chunk_size if max_duration is None else max_duration
    min_duration = chunk_size if min_duration is None else min_duration
    strict_limit_duration = chunk_size if strict_limit_duration is None else strict_limit_duration

    audio = load_audio(wav_file)
    pipeline = get_pipeline(device, vad_onset=vad_onset, vad_offset=vad_offset)
    segmentation_scores = pipeline(wav_file)
    sad_segments = Binarize(
        onset=vad_onset,
        offset=vad_offset,
        max_duration=chunk_size,
    )(segmentation_scores)

    segments: List[torch.Tensor] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0
    boundaries: List[Tuple[float, float]] = []

    def _update_segments(curr_start: float, curr_end: float, curr_duration: float):
        if curr_duration > strict_limit_duration:
            max_segments = int(curr_duration / strict_limit_duration) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
        boundaries.append((curr_start, curr_end))

    # Concat segments from pipeline into chunks for asr according to max/min duration
    # Segments longer than strict_limit_duration are split manually
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(audio.shape[0] / sr, segment.end)
        if curr_duration == 0.0:
            curr_start = start
        elif curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _update_segments(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _update_segments(curr_start, curr_end, curr_duration)

    return segments, boundaries
