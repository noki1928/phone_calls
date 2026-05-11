import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from .types import LongformTranscriptionResult, Segment, SpeakerSegment, Word


DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass
class DiarizedTranscriptionResult:
    segments: List[Segment]
    speaker_segments: List[SpeakerSegment]

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments)

    @property
    def words(self) -> List[Word]:
        result: List[Word] = []
        for segment in self.segments:
            if segment.words:
                result.extend(segment.words)
        return result

    def to_txt(self, include_timestamps: bool = False) -> str:
        lines = []
        for segment in self.segments:
            speaker = segment.speaker or "UNKNOWN"
            if include_timestamps:
                lines.append(
                    f"[{segment.start:.2f} - {segment.end:.2f}] [{speaker}] {segment.text}"
                )
            else:
                lines.append(f"[{speaker}] {segment.text}")
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        data = {
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": segment.speaker,
                    "text": segment.text,
                    "words": [
                        {
                            "start": word.start,
                            "end": word.end,
                            "speaker": word.speaker,
                            "text": word.text,
                        }
                        for word in segment.words or []
                    ],
                }
                for segment in self.segments
            ],
            "speaker_segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": segment.speaker,
                }
                for segment in self.speaker_segments
            ],
            "text": self.text,
        }
        return json.dumps(data, ensure_ascii=False, indent=indent)

    def __str__(self) -> str:
        return self.to_txt()


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_pyannote_pipeline(
    hf_token: Optional[str] = None,
    model_name: str = DEFAULT_PYANNOTE_MODEL,
    device: Optional[Union[str, torch.device]] = None,
):
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required for pyannote diarization")

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is required. Install longform extras: pip install -e '.[longform]'"
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)

    pipeline.to(_resolve_device(device))
    return pipeline


def diarize_audio(
    audio_path: str,
    hf_token: Optional[str] = None,
    pipeline=None,
    pyannote_model: str = DEFAULT_PYANNOTE_MODEL,
    device: Optional[Union[str, torch.device]] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[SpeakerSegment]:
    pipeline = pipeline or load_pyannote_pipeline(
        hf_token=hf_token,
        model_name=pyannote_model,
        device=device,
    )

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diarization = pipeline(audio_path, **kwargs)

    annotation = getattr(diarization, "speaker_diarization", diarization)
    speaker_segments = []
    try:
        iterator = annotation.itertracks(yield_label=True)
        for turn, _, speaker in iterator:
            speaker_segments.append(
                SpeakerSegment(start=turn.start, end=turn.end, speaker=str(speaker))
            )
    except AttributeError:
        for turn, speaker in annotation:
            speaker_segments.append(
                SpeakerSegment(start=turn.start, end=turn.end, speaker=str(speaker))
            )

    speaker_segments.sort(key=lambda segment: segment.start)
    return _rename_speakers(speaker_segments)


def assign_speakers(
    transcription: LongformTranscriptionResult,
    speaker_segments: List[SpeakerSegment],
) -> LongformTranscriptionResult:
    for segment in transcription.segments:
        segment.speaker = _speaker_by_overlap(segment.start, segment.end, speaker_segments)
        if not segment.words:
            continue
        for word in segment.words:
            word.speaker = _speaker_by_overlap(word.start, word.end, speaker_segments)
            if word.speaker is None:
                word.speaker = _speaker_at_time((word.start + word.end) / 2, speaker_segments)
            if word.speaker is None:
                word.speaker = segment.speaker
    return transcription


def build_speaker_turns(
    transcription: LongformTranscriptionResult,
    max_gap: float = 1.0,
    max_turn_duration: float = 60.0,
) -> List[Segment]:
    items = _word_items(transcription)
    if not items:
        items = _segment_items(transcription)

    turns: List[Segment] = []
    for item in sorted(items, key=lambda value: value["start"]):
        speaker = item["speaker"] or "UNKNOWN"
        previous = turns[-1] if turns else None
        if (
            previous
            and previous.speaker == speaker
            and item["start"] - previous.end <= max_gap
            and item["end"] - previous.start <= max_turn_duration
        ):
            previous.end = max(previous.end, item["end"])
            previous.text = _normalize_spacing(f"{previous.text} {item['text']}")
            if item.get("word") is not None:
                if previous.words is None:
                    previous.words = []
                previous.words.append(item["word"])
            continue

        words = [item["word"]] if item.get("word") is not None else None
        turns.append(
            Segment(
                text=_normalize_spacing(item["text"]),
                start=item["start"],
                end=item["end"],
                words=words,
                speaker=speaker,
            )
        )

    return turns


def transcribe_with_diarization(
    model_or_name,
    audio_path: str,
    hf_token: Optional[str] = None,
    pyannote_pipeline=None,
    pyannote_model: str = DEFAULT_PYANNOTE_MODEL,
    device: Optional[Union[str, torch.device]] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    fr_batch_size: int = 16,
    fr_num_workers: int = 0,
    merge_gap: float = 1.0,
    **longform_kwargs,
) -> DiarizedTranscriptionResult:
    if isinstance(model_or_name, str):
        from . import load_model

        model = load_model(model_or_name, device=device)
    else:
        model = model_or_name

    transcription = model.transcribe_longform(
        audio_path,
        word_timestamps=True,
        fr_batch_size=fr_batch_size,
        fr_num_workers=fr_num_workers,
        **longform_kwargs,
    )
    speaker_segments = diarize_audio(
        audio_path,
        hf_token=hf_token,
        pipeline=pyannote_pipeline,
        pyannote_model=pyannote_model,
        device=device or getattr(model, "_device", None),
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    assign_speakers(transcription, speaker_segments)
    turns = build_speaker_turns(transcription, max_gap=merge_gap)
    return DiarizedTranscriptionResult(
        segments=turns,
        speaker_segments=speaker_segments,
    )


def _rename_speakers(segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
    seen: Dict[str, str] = {}
    for segment in segments:
        if segment.speaker not in seen:
            seen[segment.speaker] = f"SPEAKER_{len(seen) + 1:02d}"
        segment.speaker = seen[segment.speaker]
    return segments


def _speaker_at_time(
    time: float,
    speaker_segments: List[SpeakerSegment],
) -> Optional[str]:
    for segment in speaker_segments:
        if segment.start <= time <= segment.end:
            return segment.speaker
    return None


def _speaker_by_overlap(
    start: float,
    end: float,
    speaker_segments: List[SpeakerSegment],
) -> Optional[str]:
    overlaps: Dict[str, float] = {}
    for segment in speaker_segments:
        overlap = max(0.0, min(end, segment.end) - max(start, segment.start))
        if overlap > 0:
            overlaps[segment.speaker] = overlaps.get(segment.speaker, 0.0) + overlap
    if not overlaps:
        return None
    return max(overlaps.items(), key=lambda item: item[1])[0]


def _word_items(transcription: LongformTranscriptionResult) -> List[dict]:
    items = []
    for segment in transcription.segments:
        for word in segment.words or []:
            text = word.text.strip()
            if text:
                items.append(
                    {
                        "start": word.start,
                        "end": word.end,
                        "speaker": word.speaker or segment.speaker,
                        "text": text,
                        "word": word,
                    }
                )
    return items


def _segment_items(transcription: LongformTranscriptionResult) -> List[dict]:
    items = []
    for segment in transcription.segments:
        text = segment.text.strip()
        if text:
            items.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": segment.speaker,
                    "text": text,
                }
            )
    return items


def _normalize_spacing(text: str) -> str:
    for punctuation in [".", ",", "!", "?", ":", ";"]:
        text = text.replace(f" {punctuation}", punctuation)
    return " ".join(text.split())


__all__ = [
    "DEFAULT_PYANNOTE_MODEL",
    "DiarizedTranscriptionResult",
    "assign_speakers",
    "build_speaker_turns",
    "diarize_audio",
    "load_pyannote_pipeline",
    "transcribe_with_diarization",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WhisperX-style GigaAM transcription with pyannote diarization"
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model-name", default="v3_e2e_rnnt")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--pyannote-model", default=DEFAULT_PYANNOTE_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--fr-batch-size", type=int, default=16)
    parser.add_argument("--fr-num-workers", type=int, default=0)
    parser.add_argument("--merge-gap", type=float, default=1.0)
    parser.add_argument("--format", choices=["txt", "json"], default="txt")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = transcribe_with_diarization(
        args.model_name,
        args.audio,
        hf_token=args.hf_token,
        pyannote_model=args.pyannote_model,
        device=args.device,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        fr_batch_size=args.fr_batch_size,
        fr_num_workers=args.fr_num_workers,
        merge_gap=args.merge_gap,
    )
    output = result.to_json() if args.format == "json" else result.to_txt(args.timestamps)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
