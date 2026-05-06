from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional
import tempfile
import os

from app.whisper_service import WhisperService
from app.summarizer import SummarizerService


whisper_service: Optional[WhisperService] = None
summarizer_service: Optional[SummarizerService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_service, summarizer_service
    print("Initializing Whisper service...")
    whisper_service = WhisperService()
    whisper_service.initialize()
    
    print("Initializing Summarizer service...")
    summarizer_service = SummarizerService()
    
    print("Services initialized")
    yield
    
    print("Shutting down...")


app = FastAPI(title="Whisper + Summarizer API", lifespan=lifespan)


@app.post("/transcribe-and-summarize")
async def transcribe_and_summarize(
    file: UploadFile = File(..., description="WAV audio file"),
    custom_prompt: Optional[str] = Form(None, description="Optional custom system prompt for summarization")
):
    if not file.filename.lower().endswith(('.wav', '.wave')):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        transcription = whisper_service.transcribe(tmp_path)
        summary = summarizer_service.summarize(transcription, custom_prompt)

        return JSONResponse(content={
            "transcription": transcription,
            "summary": summary
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8096)