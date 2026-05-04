from datetime import timedelta, datetime
from typing import Optional
import os
import uuid

from fastapi import FastAPI, Request, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import ffmpeg
import numpy as np
import srt as srt
import stable_whisper
from deep_translator import GoogleTranslator

from models import SessionLocal, AudioFile

DEFAULT_MAX_CHARACTERS = 80


def get_audio_buffer(filename: str, start: int, length: int):
    """
    input: filename of the audio file, start time in seconds, length of the audio in seconds
    output: np array of the audio data which the model's transcribe function can take as input
    """
    out, _ = (
        ffmpeg.input(filename, threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000, ss=start, t=length)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def transcribe_time_stamps(segments: list):
    """
    input: a list of segments from the model's transcribe function
    output: a string of the timestamps and the text of each segment
    """
    string = ""
    for seg in segments:
        string += " ".join([str(seg.start), "->", str(seg.end), ": ", seg.text.strip(), "\n"])
    return string


def split_text_by_punctuation(text: str, max_length: int):
    chunks = []
    while len(text) > max_length:

        split_pos = max(
            (text.rfind(p, 0, max_length) for p in [",", ".", "?", "!"," "] if p in text[:max_length]),
            default=-1
        )


        if split_pos == -1:
            split_pos = max_length


        chunks.append(text[:split_pos + 1].strip())
        text = text[split_pos + 1:].strip()

    if text:
        chunks.append(text)

    return chunks


def translate_text(text: str, translate_to: str):
    return GoogleTranslator(source='auto', target=translate_to).translate(text=text)


def make_srt_subtitles(segments: list, translate_to: str, max_chars: int):
    subtitles = []
    for i, seg in enumerate(segments, start=1):
        start_time = seg.start
        end_time = seg.end
        text = translate_text(seg.text.strip(), translate_to)

        text_chunks = split_text_by_punctuation(text, max_chars)

        duration = (end_time - start_time) / len(text_chunks)

        for j, chunk in enumerate(text_chunks):
            chunk_start = start_time + j * duration
            chunk_end = chunk_start + duration

            subtitle = srt.Subtitle(
                index=len(subtitles) + 1,
                start=timedelta(seconds=chunk_start),
                end=timedelta(seconds=chunk_end),
                content=chunk
            )
            subtitles.append(subtitle)

    return srt.compose(subtitles)


appold = FastAPI(debug=True)

appold.mount('/static', StaticFiles(directory='static'), name='static')
template = Jinja2Templates(directory='templates')


@appold.get('/', response_class=HTMLResponse)
def index(request: Request):
    db = SessionLocal()
    try:
        audio_files = db.query(AudioFile).order_by(AudioFile.created_at.desc()).all()
        return template.TemplateResponse('index.html', {"request": request, "text": None, "audio_files": audio_files})
    finally:
        db.close()


@appold.post('/download/')
async def download_subtitle(
        request: Request,
        file: bytes = File(),
        model_type: str = Form("tiny"),
        timestamps: Optional[str] = Form("False"),
        filename: str = Form("subtitles"),
        file_type: str = Form("srt"),
        max_characters: int = Form(DEFAULT_MAX_CHARACTERS),
        translate_to: str = Form('spanish'),
):
    db = SessionLocal()
    
    # Create unique filename
    unique_id = str(uuid.uuid4())
    audio_filename = f"audio_{unique_id}.mp3"
    subtitle_filename = f"subtitle_{unique_id}.{file_type}"
    
    # Create uploads and subtitles directories if they don't exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('subtitles', exist_ok=True)
    
    # Create database record
    audio_file = AudioFile(
        filename=subtitle_filename,
        original_name=filename,
        status="processing",
        model_type=model_type,
        translation_language=translate_to,
        subtitle_format=file_type
    )
    db.add(audio_file)
    db.commit()

    try:
        # Save audio file
        with open(f'uploads/{audio_filename}', 'wb') as f:
            f.write(file)
        
        # Process audio
        model = stable_whisper.load_model(model_type)
        result = model.transcribe(f'uploads/{audio_filename}', regroup=False)

        # Generate subtitles
        subtitle_path = f'subtitles/{subtitle_filename}'
        with open(subtitle_path, 'w') as f:
            if timestamps == "True":
                f.write(make_srt_subtitles(result.segments, translate_to, max_characters))
            else:
                f.write(result.text)

        # Update database record
        audio_file.status = "completed"
        audio_file.completed_at = datetime.utcnow()
        db.commit()

        # Return file for download
        return FileResponse(
            subtitle_path,
            media_type="application/octet-stream",
            filename=f"{filename}.{file_type}"
        )

    except Exception as e:
        audio_file.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@appold.get('/download/{file_id}')
def download_processed(file_id: int):
    db = SessionLocal()
    try:
        audio_file = db.query(AudioFile).filter_by(id=file_id).first()
        if not audio_file:
            raise HTTPException(status_code=404, detail="File not found")
        
        if audio_file.status != "completed":
            raise HTTPException(status_code=400, detail="File not ready")
            
        return FileResponse(
            f'subtitles/{audio_file.filename}',
            media_type="application/octet-stream",
            filename=f"{audio_file.original_name}.{audio_file.subtitle_format}"
        )
    finally:
        db.close()
