from io import BytesIO
import os
import time
from typing import Union
from fastapi import Form, File, Depends, BackgroundTasks
from starlette.datastructures import UploadFile
import json
import numpy as np
from ...logging import logger

from .types import BaseFilePayload, validate_file_payload

from ..model_binding import get_MODEL_EMBEDDING, get_MODEL_ASR, \
    CONFIG_ASR, standardize_audio_format, get_transcription_segments, get_duration, get_diarization

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
        logger.info(payload)
        _file = payload.file_name
        
        if isinstance(_file, UploadFile):
            _audio = BytesIO(_file.file.read())
            
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
            
def shell_func(audio, language, if_diarize):
    audio_np: np.ndarray = standardize_audio_format(audio)
    duration: float = get_duration(audio=audio_np, sample_rate=16000)
    
    transcript, info = get_transcription_segments(
        model_asr=MODEL_ASR,
        config_asr=CONFIG_ASR,
        audio=audio_np,
        language=language,
        generation_mode=False,
        return_info=True,
    )
    print(if_diarize)
    print(transcript)
    if if_diarize:
        transcript = get_diarization(
            model_embedding=MODEL_EMBEDDING,
            audio=audio_np,
            segments=transcript,
            return_type="dict"
        )
    
    return transcript, info
