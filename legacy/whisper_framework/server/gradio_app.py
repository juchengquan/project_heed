import os
import asyncio
import json
import time
import random
import gradio as gr
import pandas as pd
import torch

from ..models import asr_model, embedding_model, tokenizer, model
# from ..models import asr_model, embedding_model

from ..functions import make_embeddings, get_num_speakers, make_output, convert_wav
from ..functions import get_fked

def func_clean():
    return gr.DataFrame.update(value=None), gr.State.update(value=None), gr.State.update(value=None)
    
def func_transcribe(AUDIO_FILE):
    print(AUDIO_FILE)
    # AUDIO_FILE = convert_wav(AUDIO_FILE)
    # make_embeddings
    # Transcribe audio
    options = dict(language="en", beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)
    # lazy execution
    segments_raw, info = asr_model.transcribe(AUDIO_FILE, **transcribe_options, vad_filter=True)

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    # Convert back to original openai format
    segments = []
    i = 0
    for segment_chunk in segments_raw:
        chunk = {}
        chunk["start"] = segment_chunk.start
        chunk["end"] = segment_chunk.end
        chunk["text"] = segment_chunk.text
        segments.append(chunk)
        i += 1
    print("transcribe audio done with fast whisper")

    _embeddings = make_embeddings(AUDIO_FILE, segments, embedding_model)

    for _ in range(3):
        torch.cuda.empty_cache()

    segments = get_num_speakers(segments, _embeddings)
    outputs = make_output(segments)
    result = pd.DataFrame(outputs)
    return result

def get_diarization(AUDIO_FILE, ts_state_chunk):
    _embeddings = make_embeddings(AUDIO_FILE, ts_state_chunk, embedding_model)

    for _ in range(3):
        torch.cuda.empty_cache()

    segments = get_num_speakers(ts_state_chunk, _embeddings)
    outputs = make_output(segments)
    result = pd.DataFrame(outputs)
    return result

def get_transcription_segments(AUDIO_FILE, ts_state, ts_state_chunk):
    if not ts_state:
        ts_state = []
    if not ts_state_chunk:
        ts_state_chunk = []
        
    AUDIO_FILE = convert_wav(AUDIO_FILE)
    # Transcribe audio
    options = dict(language="en", beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)
    # lazy execution
    segments_raw, _ = asr_model.transcribe(AUDIO_FILE, **transcribe_options, vad_filter=True)

    # segments = []
    cnt = 0
    for segment_chunk in segments_raw:
        ts_state.append([segment_chunk.start, segment_chunk.end, "", segment_chunk.text])
        cnt += 1
        chunk = {
            "start": segment_chunk.start,
            "end": segment_chunk.end,
            "text": segment_chunk.text,

        }
        ts_state_chunk.append(chunk)
        if cnt >= 5:
            yield ts_state, ts_state_chunk
            cnt = 0

    yield ts_state, ts_state_chunk
    for _ in range(3):
        torch.cuda.empty_cache()
    # print("transcribe audio done with fast whisper")

def func_summarize(transcript):
    return get_fked(transcript, tokenizer, model)

with gr.Blocks(theme=gr.themes.Base()) as gr_app:
    gr_app.ssl_verify = False

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(source="upload", type="filepath")
        with gr.Column(scale=2):
            btn_clean = gr.Button("Clean")
            # btn_transcribe = gr.Button("Transcribe")
            # btn_diarize = gr.Button("Diarize")
            btn_summarize = gr.Button("Summarize")

    text_summ = gr.Textbox(label='Summarization')
    transcript = gr.DataFrame(headers=["start", "end", "speaker", "text"], interactive=False)
    ts_state = gr.State()
    ts_state_chunk = gr.State()

    btn_clean.click(func_clean, inputs=[], outputs=[transcript, ts_state, ts_state_chunk])

    btn_summarize \
        .click(
            func_transcribe, 
            inputs=[audio], 
            outputs=[transcript],
            # queue=True,
        ) \
        .then(
            func_summarize, inputs=[transcript], outputs=[text_summ]
        ) 

# gr_app = gr_app.queue(concurrency_count=1) # concurrency_count=2)


if __name__ == "__main__":
    gr_app.launch(ssl_verify=False)
