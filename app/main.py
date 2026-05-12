"""FastAPI application for GigaAM transcription and dialog summarization."""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

from app.gigaam_service import GigaAMService
from app.summarizer import SummarizerService


gigaam_service: Optional[GigaAMService] = None
summarizer_service: Optional[SummarizerService] = None


class SystemPromptUpdate(BaseModel):
    """Request body for updating the summarization system prompt."""

    system_prompt: str


class ModelUpdate(BaseModel):
    """Request body for updating the summarization model name."""

    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize long-lived transcription and summarization services."""

    global gigaam_service, summarizer_service
    print("Initializing GigaAM service...")
    gigaam_service = GigaAMService()
    gigaam_service.initialize()
    
    print("Initializing Summarizer service...")
    summarizer_service = SummarizerService()
    
    print("Services initialized")
    yield
    
    print("Shutting down...")


app = FastAPI(title="GigaAM + Summarizer API", lifespan=lifespan)


@app.post("/transcribe-and-summarize")
async def transcribe_and_summarize(
    file: UploadFile = File(..., description="WAV audio file")
):
    """Transcribe an uploaded WAV file with GigaAM and return its summary."""

    if not file.filename.lower().endswith(('.wav', '.wave')):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        transcription = gigaam_service.transcribe(tmp_path)
        summary = summarizer_service.summarize(transcription)

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
    """Return service health status."""

    return {"status": "healthy"}


@app.get("/summarization/system-prompt")
async def get_summarization_system_prompt():
    """Return the current summarization system prompt."""

    return {"system_prompt": summarizer_service.get_system_prompt()}


@app.put("/summarization/system-prompt")
async def set_summarization_system_prompt(update: SystemPromptUpdate):
    """Update the summarization system prompt."""

    try:
        system_prompt = summarizer_service.set_system_prompt(update.system_prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"system_prompt": system_prompt}


@app.get("/summarization/model")
async def get_summarization_model():
    """Return the current summarization model name."""

    return {"model": summarizer_service.get_model()}


@app.put("/summarization/model")
async def set_summarization_model(update: ModelUpdate):
    """Update the summarization model name."""

    try:
        model = summarizer_service.set_model(update.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"model": model}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8096)
