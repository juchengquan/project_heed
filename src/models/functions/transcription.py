from io import BytesIO
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Union
import torch

from .misc import convert_time
from ...logging import logger
from ..faster_whisper import MODEL_CONFIG, CONFIG_ASR

def _transcribe(
        model_asr,
        config_asr,
        audio: Union[BytesIO, str],
        config_manual: dict = {}
    ):
    _config_asr = config_asr.copy()
    _config_asr.update(config_manual)
    
    logger.info(_config_asr)
    # Transribe audio
    transcribe_options = dict(task="transcribe", vad_filter=True, **_config_asr)
    
    segments, info = model_asr.transcribe(
        audio = audio,
        **transcribe_options
    )
    return segments, info

def _get_segment_by_language(
        segment_raw,
        generation_mode: bool=True,
        yield_speed: int=5,
        language: str = "en",
    ):
    pat_sentence_en = re.compile(r".*?[^\w\d,:]$")
    
    cnt = 0
    tmp_text = ""
    _segments: list = []
    
    if language == "en":
        tmp_start = -1
        for segment_chunk in segment_raw:
            if tmp_text == "":
                tmp_start = segment_chunk.start
            tmp_text += segment_chunk.text  
            if re.search(pattern=pat_sentence_en, string=segment_chunk.text):
                tmp_text = tmp_text.strip(" ")
                _segments.append({
                    "start": tmp_start,
                    "end": segment_chunk.end,
                    "text": tmp_text,
                })
                
                tmp_text = ""
                tmp_start = segment_chunk.end
                cnt +=1
            
            if cnt >= yield_speed:
                if generation_mode:
                    yield _segments
                cnt = 0
        if tmp_text:
            _segments.append({
                "start": tmp_start,
                "end": segment_chunk.end,
                "text": tmp_text,
            })
    elif language == "zh":
        for segment_chunk in segment_raw:
            tmp_text = tmp_text.strip(" ")
            _segments.append({
                "start": tmp_start,
                "end": segment_chunk.end,
                "text": tmp_text,
            })
            cnt += 1
            if cnt >= yield_speed:
                if generation_mode:
                    yield _segments
                cnt = 0
    else:
        raise NotImplementedError("Language not implemented.")

    torch.cuda.empty_cache()
    yield _segments
            
def get_transcription_segments(
        model_asr,
        config_asr,
        audio: Union[BytesIO, str],
        language: str = "",
        generation_mode: bool = True,
        yield_speed: int = 5,
        return_info: bool = False,
    ):
    if language:
        segments_raw, info = _transcribe(model_asr=model_asr, config_asr=config_asr, audio=audio, config_manual={"language": language})
    else:
        segments_raw, info = _transcribe(model_asr=model_asr, config_asr=config_asr, audio=audio, config_manual={"language": "en"})
    
    _lang = info.language if info.language in ["en", "zh"] else "en"
    res: list = _get_segment_by_language(
        segment_raw=segments_raw, generation_mode=generation_mode, yield_speed=yield_speed, language=_lang
    )
    res_temp = list(res)
    print(res_temp)
    _segments = res_temp[-1]
    
    if isinstance(audio, BytesIO):
        audio.seek(0)
    logger.info("Transcription done.")
    return _segments, info if return_info else _segments
