"""Shared typed containers for GigaAM transcription results."""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from torch import Tensor


@dataclass
class AudioDatasetSample:
    """Audio sample and optional reference text or tokens for batching."""

    item: Union[str, np.ndarray, Tensor]
    duration: float
    text: Optional[str] = None
    tokens: Optional[List[int]] = None


@dataclass
class Word:
    """Word-level transcription item with optional speaker label."""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Plain transcription result with optional word timestamps."""

    text: str
    words: Optional[List[Word]] = None

    def __str__(self) -> str:
        """Return transcript text."""

        return self.text


@dataclass
class Segment:
    """Time-bounded transcript segment with optional words and speaker label."""

    text: str
    start: float
    end: float
    words: Optional[List[Word]] = None
    speaker: Optional[str] = None


@dataclass
class SpeakerSegment:
    """Time-bounded diarization segment with a speaker label."""

    start: float
    end: float
    speaker: str


@dataclass
class LongformTranscriptionResult:
    """Long-form transcription result split into timestamped segments."""

    segments: List[Segment]

    @property
    def words(self) -> List[Word]:
        """Flatten all words from all segments."""
        result = []
        for seg in self.segments:
            if seg.words:
                result.extend(seg.words)
        return result

    @property
    def has_word_timestamps(self) -> bool:
        """Return whether segments contain word-level timestamps."""

        return bool(self.segments) and self.segments[0].words is not None

    @property
    def text(self) -> str:
        """Return the full transcript text."""

        return " ".join(s.text for s in self.segments)

    def __str__(self) -> str:
        """Return transcript text."""

        return self.text

    def __iter__(self):
        """Iterate over transcript segments."""

        return iter(self.segments)

    def __len__(self) -> int:
        """Return the number of transcript segments."""

        return len(self.segments)
