"""HTTP API for bounded, single-image deepfake inference."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import warnings
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .inference import predict_image


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))
ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
DEFAULT_ORIGINS = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000"
ALLOW_ORIGINS = [origin.strip() for origin in os.getenv("ALLOW_ORIGINS", DEFAULT_ORIGINS).split(",") if origin.strip()]
API_KEY = os.getenv("API_KEY")

if MAX_UPLOAD_BYTES <= 0 or MAX_IMAGE_PIXELS <= 0 or MAX_CONCURRENT_INFERENCES <= 0:
    raise RuntimeError("Upload, image, and inference limits must be positive.")

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)


class PredictionResponse(BaseModel):
    label: Literal["REAL", "FAKE"]
    decision_score: float = Field(ge=0, le=1)
    real_probability: float = Field(ge=0, le=1)
    fake_probability: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)


app = FastAPI(
    title="Deepfake Detection API",
    description="Single-face image classification. Results are model estimates, not proof of authenticity.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    if request.url.path == "/predict" and API_KEY:
        supplied_key = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(supplied_key, API_KEY):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key."},
                headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path == "/predict":
        response.headers["Cache-Control"] = "no-store"
    return response


async def _read_limited_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")

    content = bytearray()
    while chunk := await file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Image must be {MAX_UPLOAD_BYTES // (1024 * 1024)} MB or smaller.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return bytes(content)


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format")
            source.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            return image.copy()
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid supported image.") from error


@app.get("/")
def root():
    return {"message": "Deepfake Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image = _decode_image(await _read_limited_upload(file))
        try:
            async with inference_semaphore:
                result = await run_in_threadpool(predict_image, image)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected prediction failure")
        raise HTTPException(status_code=500, detail="Unable to process the image.") from None
    finally:
        await file.close()


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers={"Cache-Control": "no-store"})
