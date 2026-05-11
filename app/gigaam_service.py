import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import gigaam

from app.config import settings


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


class GigaAMService:
    def __init__(self):
        self.device = settings.gigaam.device
        self.model_name = settings.gigaam.model_name
        self.pyannote_model = settings.gigaam.pyannote_model
        self.hf_token = settings.hf_token
        self.num_speakers = settings.gigaam.num_speakers
        self.min_speakers = settings.gigaam.min_speakers
        self.max_speakers = settings.gigaam.max_speakers
        self.fr_batch_size = settings.gigaam.fr_batch_size
        self.fr_num_workers = settings.gigaam.fr_num_workers
        self.merge_gap = settings.gigaam.merge_gap

        self.model = None
        self.diarize_model = None

    def initialize(self):
        print(f"Loading GigaAM model: {self.model_name}")
        self.model = gigaam.load_model(self.model_name, device=self.device)

        print(f"Loading pyannote diarization model: {self.pyannote_model}")
        self.diarize_model = self._load_diarization_model()

        print("GigaAM and pyannote models loaded")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None or self.diarize_model is None:
            raise RuntimeError("GigaAM service not initialized")

        transcription = self.model.transcribe_longform(
            audio_path,
            word_timestamps=True,
            fr_batch_size=self.fr_batch_size,
            fr_num_workers=self.fr_num_workers,
        )
        speaker_segments = self._diarize(audio_path)
        self._assign_speakers(transcription, speaker_segments)
        turns = self._build_speaker_turns(transcription)

        return "\n".join(f"[{turn['speaker']}] {turn['text']}" for turn in turns)

    def _load_diarization_model(self):
        if not self.hf_token:
            raise RuntimeError("HF_TOKEN is required for pyannote diarization")

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio is not installed. Install dependencies from requirements.txt"
            ) from exc

        try:
            pipeline = Pipeline.from_pretrained(
                self.pyannote_model,
                token=self.hf_token,
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                self.pyannote_model,
                use_auth_token=self.hf_token,
            )

        pipeline.to(torch.device(self.device))
        return pipeline

    def _diarize(self, audio_path: str) -> List[SpeakerSegment]:
        kwargs = {}
        if self.num_speakers is not None:
            kwargs["num_speakers"] = self.num_speakers
        else:
            if self.min_speakers is not None:
                kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                kwargs["max_speakers"] = self.max_speakers

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diarization = self.diarize_model(audio_path, **kwargs)

        annotation = getattr(diarization, "speaker_diarization", diarization)
        segments = []
        try:
            iterator = annotation.itertracks(yield_label=True)
            for turn, _, speaker in iterator:
                segments.append(
                    SpeakerSegment(turn.start, turn.end, str(speaker))
                )
        except AttributeError:
            for turn, speaker in annotation:
                segments.append(
                    SpeakerSegment(turn.start, turn.end, str(speaker))
                )

        segments.sort(key=lambda item: item.start)
        return self._rename_speakers(segments)

    def _assign_speakers(self, transcription, speaker_segments: List[SpeakerSegment]) -> None:
        for segment in transcription:
            segment.speaker = self._speaker_by_overlap(
                segment.start,
                segment.end,
                speaker_segments,
            )
            for word in segment.words or []:
                word.speaker = self._speaker_by_overlap(
                    word.start,
                    word.end,
                    speaker_segments,
                )
                if word.speaker is None:
                    word.speaker = segment.speaker or "UNKNOWN"

    def _build_speaker_turns(self, transcription) -> List[dict]:
        items = []
        for segment in transcription:
            if segment.words:
                for word in segment.words:
                    text = word.text.strip()
                    if text:
                        items.append({
                            "start": word.start,
                            "end": word.end,
                            "speaker": getattr(word, "speaker", None) or getattr(segment, "speaker", None) or "UNKNOWN",
                            "text": text,
                        })
            else:
                text = segment.text.strip()
                if text:
                    items.append({
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": getattr(segment, "speaker", None) or "UNKNOWN",
                        "text": text,
                    })

        turns = []
        for item in sorted(items, key=lambda value: value["start"]):
            previous = turns[-1] if turns else None
            if (
                previous
                and previous["speaker"] == item["speaker"]
                and item["start"] - previous["end"] <= self.merge_gap
            ):
                previous["end"] = max(previous["end"], item["end"])
                previous["text"] = self._normalize_spacing(
                    f"{previous['text']} {item['text']}"
                )
                continue

            turns.append({
                "start": item["start"],
                "end": item["end"],
                "speaker": item["speaker"],
                "text": self._normalize_spacing(item["text"]),
            })

        return turns

    @staticmethod
    def _rename_speakers(segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        speaker_map = {}
        for segment in segments:
            if segment.speaker not in speaker_map:
                speaker_map[segment.speaker] = f"SPEAKER_{len(speaker_map) + 1:02d}"
            segment.speaker = speaker_map[segment.speaker]
        return segments

    @staticmethod
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

    @staticmethod
    def _normalize_spacing(text: str) -> str:
        for punctuation in [".", ",", "!", "?", ":", ";"]:
            text = text.replace(f" {punctuation}", punctuation)
        return " ".join(text.split())
