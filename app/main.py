"""Point d'entrée du service réceptionniste IA."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog

from app.config import config
from app.ari_handler import ari_handler

# Configuration du logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    logger.info("starting_receptionniste", company=config.company.name)

    # Démarrer le handler ARI en arrière-plan
    ari_task = asyncio.create_task(ari_handler.start())

    yield

    # Arrêter proprement
    logger.info("stopping_receptionniste")
    ari_task.cancel()
    await ari_handler.stop()


app = FastAPI(
    title="TKSA Réceptionniste IA",
    description="Standard téléphonique IA pour Toni Küpfer SA",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "service": "TKSA Réceptionniste IA",
        "company": config.company.name,
        "status": "running",
        "services": [
            {"name": s.name, "extension": s.extension}
            for s in config.company.services
        ]
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


class TestTTSRequest(BaseModel):
    text: str


@app.post("/test/tts")
async def test_tts(request: TestTTSRequest):
    """Teste la génération TTS."""
    from app.ai_handler import ai_handler

    try:
        audio_path = await ai_handler.text_to_speech(request.text)
        return {"status": "ok", "audio_path": audio_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TestSTTRequest(BaseModel):
    audio_path: str


@app.post("/test/stt")
async def test_stt(request: TestSTTRequest):
    """Teste la transcription STT."""
    from app.ai_handler import ai_handler

    try:
        transcript = await ai_handler.speech_to_text(request.audio_path)
        return {"status": "ok", "transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TestIntentRequest(BaseModel):
    text: str


@app.post("/test/intent")
async def test_intent(request: TestIntentRequest):
    """Teste la compréhension d'intention."""
    from app.ai_handler import ai_handler

    try:
        result = await ai_handler.understand_intent(request.text)
        return {
            "status": "ok",
            "service": result["service"].name if result["service"] else None,
            "response": result["response"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """Liste les sessions d'appel en cours."""
    return {
        "count": len(ari_handler.sessions),
        "sessions": [
            {
                "channel_id": s.channel_id,
                "caller_id": s.caller_id,
                "state": s.state.value,
                "target_service": s.target_service.name if s.target_service else None
            }
            for s in ari_handler.sessions.values()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
