import time
import numpy as np
from io import BytesIO
from fastapi import Depends
from starlette.datastructures import UploadFile

from .types import BaseFilePayload, validate_file_payload

from ..model_binding import get_MODEL_EMBEDDING, get_MODEL_ASR, CONFIG_ASR, \
    standardize_audio_format, get_transcription_segments, get_duration, get_diarization

MODEL_EMBEDDING = get_MODEL_EMBEDDING()
MODEL_ASR = get_MODEL_ASR()

async def api_health():
    return {"text": "OK"}

async def api_transribe_with_file(
        payload: BaseFilePayload = Depends()
    ):
    try:
        t_s = time.time()
        payload = validate_file_payload(payload=payload)
        _file = payload.file_name
        
        if isinstance(_file, UploadFile):
            _audio: BytesIO = BytesIO(_file.file.read())
            
            transcript, info = shell_func(_audio, payload.language, payload.if_diarize)
            
            return {
                "transcripts": transcript,
                "language": info.language,
                "language_probability": info.language_probability,
                "trace_id": payload.trace_id,
                "elapsed_time": time.time() - t_s,
            }
        else:
            raise NotImplementedError(f"Not implemented: {type(_file)}")
    except Exception as err:
        raise err
        return {
            "error": str(err)
        }
            
def shell_func(
        audio: BytesIO,
        language: str, 
        if_diarize: str,
    ):
    audio_np: np.ndarray = standardize_audio_format(audio)
    duration: float = get_duration(audio=audio_np, sample_rate=16000)
    
    transcript, info = get_transcription_segments(
        audio=audio_np,
        model_asr=MODEL_ASR,
        config_asr=CONFIG_ASR,
        language=language,
        generation_mode=False,
        return_info=True,
    )
    if if_diarize:
        transcript = get_diarization(
            audio=audio_np,
            segments=transcript,
            model_embedding=MODEL_EMBEDDING,
            return_type="dict"
        )
    
    return transcript, info
