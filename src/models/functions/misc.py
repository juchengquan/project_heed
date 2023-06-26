import io
from io import BytesIO
import datetime
import numpy as np
import pandas as pd
import warnings
# import soundfile as sf

from typing import List, Dict, Union, Tuple
from .audio import decode_audio

# def convert_wav(path_audio_file):
#     data, samplerate = sf.read(path_audio_file)
#     sf.write(path_audio_file, data, samplerate, subtype='PCM_16')
#     return path_audio_file

def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

def standardize_audio_format(audio_content: BytesIO):
    res_np: np.array = decode_audio(audio_content, sampling_rate=16000, split_stereo=False)
    audio_content.seek(0)
    return res_np

def get_duration(audio: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], str], sample_rate: int = 16000) -> float:
    if isinstance(audio, np.ndarray):
        duration = audio.shape[0] / sample_rate
    elif isinstance(audio, Tuple):
        duration = audio[0].shape[0] / sample_rate
        raise NotImplementedError("Does not yet support split_stereo.")
    elif isinstance(audio, str):
        import wave
        # with contextlib.closing(wave.open(path_audio_file,'r')) as f:
        with wave.open(audio, "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / rate
    else:
        raise NotImplementedError("Type not supported.")
    return duration



def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    breakpoint()
    if data.dtype == np.float32:
        warnings.warn(
            "Audio data is not in 16-bit integer format."
            "Trying to convert to 16-bit int format."
        )
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        warnings.warn(
            "Audio data is not in 16-bit integer format."
            "Trying to convert to 16-bit int format."
        )
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint8:
        warnings.warn(
            "Audio data is not in 16-bit integer format."
            "Trying to convert to 16-bit int format."
        )
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError("Audio data cannot be converted to " "16-bit int format.")
    return data
