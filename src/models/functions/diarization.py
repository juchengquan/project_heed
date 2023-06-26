import io
import wave
import warnings
import torch
import numpy as np
import pandas as pd

from typing import List, Dict, Union, Tuple

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.io import wavfile

from .misc import convert_time, get_duration

from pyannote.audio import Audio
from pyannote.core import Segment

from ...logging import logger

from ..pretrained.speechbrain_pretrained_speakerembedding import SpeechBrainPretrainedSpeakerEmbedding

def _get_diarization(
        segments: List[Dict], 
        embeddings: np.ndarray, 
        min_no: int = 2, 
        max_no: int = 5,
    ) -> List[Dict]:
    num_speakers = 0

    if num_speakers == 0:
    # Find the best number of speakers
        score_num_speakers = {}

        for num_speakers in range(min_no, max_no+1):
            try:
                clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_, metric="euclidean")
                score_num_speakers[num_speakers] = score
            except:
                continue
        
        # pre-caution
        if score_num_speakers == {}:
            score_num_speakers[1] = 666
            
        best_num_speaker = max(score_num_speakers, key=lambda x:score_num_speakers[x])
        logger.info(f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")
    else:
        best_num_speaker = num_speakers
        logger.info(f"The best number of speakers: {best_num_speaker}")

    # Assign speaker label   
    clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)
        
    return segments

def _make_embeddings(
        model_embedding: SpeechBrainPretrainedSpeakerEmbedding,
        audio: Union[np.ndarray, io.BytesIO, str],
        segments: List[Dict],
    ):
    duration: float = get_duration(audio)
    logger.info(f"Duration: {duration}")
    
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = _segment_embedding(
            model_embedding=model_embedding,
            audio=audio,
            segment=segment,
            duration=duration
        )
    torch.cuda.empty_cache()
    return np.nan_to_num(embeddings)

# Create embedding
def _segment_embedding(
        model_embedding: SpeechBrainPretrainedSpeakerEmbedding,
        audio: Union[np.ndarray, str, io.BytesIO],
        segment: Dict,
        duration: float,
    ) -> np.ndarray:
    if isinstance(audio, str):
        audio = Audio()
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, _ = audio.crop(audio, clip)
        return model_embedding(waveform[None])

    if isinstance(audio, np.ndarray):
        audio_obj = Audio()
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        wtf_in = {
            "waveform": torch.from_numpy(np.expand_dims(audio, axis=0)),
            "sample_rate": 16000,
        }
        waveform, _ = audio_obj.crop(wtf_in, clip)
        return model_embedding(waveform[None])
    else:
        raise NotImplementedError("Not implemented im '_segment_embedding'.")

def get_diarization(
        model_embedding: SpeechBrainPretrainedSpeakerEmbedding,
        audio: str,
        segments: List[Dict],
        return_type: str = "list"
    ) -> Union[List[List], List[Dict]]:
    _embeddings: np.ndarray = _make_embeddings(model_embedding=model_embedding, audio=audio, segments=segments)
    
    segments: List[Dict] = _get_diarization(segments=segments, embeddings=_embeddings)
    
    outputs: pd.DataFrame = _make_diarization_outputs (segments=segments)
    
    if return_type == "list":
        return outputs.to_dict(orient="split")["data"]
    elif return_type == "dict":
        return outputs.to_dict(orient="records")
    else:
        raise NotImplementedError("Not implemented")

# Make output
def _make_diarization_outputs(segments: List[Dict]) -> pd.DataFrame:
    objects = {
        "start" : [],
        "end": [],
        "speaker": [],
        "text": []
    }
    text = ""
    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            objects["start"].append(str(convert_time(segment["start"])))
            objects["speaker"].append(segment["speaker"])
            if i != 0:
                objects["end"].append(str(convert_time(segments[i - 1]["end"])))
                objects["text"].append(text)
                text = ""
        text += segment["text"] + " "
    if text:
        objects["end"].append(str(convert_time(segments[i]["end"]))) # TODO
        objects["text"].append(text)
    
    df_obj = pd.DataFrame(objects)
    
    return df_obj