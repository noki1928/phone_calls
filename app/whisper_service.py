import whisperx
from whisperx.diarize import DiarizationPipeline

from app.config import settings


class WhisperService:
    def __init__(self):
        self.device = settings.whisper.device
        self.model_size = settings.whisper.model_size
        self.compute_type = settings.whisper.compute_type
        self.language = settings.whisper.language
        self.hf_token = settings.hf_token
        self.num_speakers = settings.whisper.num_speakers
        self.min_speakers = settings.whisper.min_speakers
        self.max_speakers = settings.whisper.max_speakers

        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

    def initialize(self):
        print(f"Loading Whisper model: {self.model_size}")
        self.model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type
        )
        
        print("Loading align model...")
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=self.language,
            device=self.device
        )
        
        print("Loading diarize model...")
        self.diarize_model = DiarizationPipeline(token=self.hf_token, device=self.device)
        
        print("All Whisper models loaded")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            raise RuntimeError("Whisper service not initialized")

        audio = whisperx.load_audio(audio_path)
        
        result = self.model.transcribe(audio, language=self.language)
        
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device
        )

        diarize_segments = self.diarize_model(
            audio,
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers
        )

        result = whisperx.assign_word_speakers(diarize_segments, result)

        final_string = ""
        for seg in result["segments"]:
            speaker = seg.get("speaker", "UNKNOWN")
            final_string += f"[{speaker}] {seg['text']}\n"

        return final_string