import os
import asyncio
import json
import time
import random
import gradio as gr
import pandas as pd
import torch

from ..models import asr_model, embedding_model, tokenizer, model
from ..functions import make_embeddings, get_num_speakers, make_outputs, convert_wav
from ..functions import get_fked

from ..functions.funcs_gradio import get_transcription_segments, get_diarization

def func_clean():
    return gr.DataFrame.update(value=None), gr.State.update(value=None), gr.State.update(value=None)

def get_summarize(transcript):
    return get_fked(transcript, tokenizer, model)

def func_summarize(transcript):
    return get_fked(transcript, tokenizer, model)

with gr.Blocks(theme=gr.themes.Base()) as gr_app:
    gr_app.ssl_verify = False

    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(source="upload", type="filepath")
        with gr.Column(scale=2):
            btn_clean = gr.Button("Clean")
            btn_transcribe = gr.Button("Transcribe")
            btn_diarize = gr.Button("Diarize")
            btn_summarize = gr.Button("Summarize")

    text_summ = gr.Textbox(label='Summarization')
    transcript = gr.DataFrame(headers=["start", "end", "speaker", "text"], interactive=False)
    ts_state = gr.State()
    ts_state_chunk = gr.State()

    btn_transcribe.click(
        get_transcription_segments, 
        asr_model,
        inputs=[audio, ts_state, ts_state_chunk], 
        outputs=[transcript, ts_state_chunk], 
        queue=True
    )

    btn_diarize.click(
        get_diarization, 
        inputs=[audio, ts_state_chunk], 
        outputs=[transcript],
        queue=True,
    )

    btn_summarize.click(
        func_summarize, 
        inputs=[transcript], 
        outputs=[text_summ],
        queue=True,
    )

    btn_clean.click(func_clean, inputs=[], outputs=[transcript, ts_state, ts_state_chunk])

    # btn_summarize\
    #     .click(
    #         func_summarize, inputs=[transcript], outputs=[text_summ]
    #     )


    # with gr.Row().style(equal_height=True):
    #     with gr.Column(scale=4):
    #         with gr.Row():
    #             text_input = gr.Textbox(label="Context", placeholder="Context:", lines=10, visible=True)
    #         with gr.Row():
    #             btn_clean = gr.Button("Clean")
    #             btn_transcribe = gr.Button("Submit")

    #     with gr.Column(scale=4):
    #         with gr.Row():
    #             # with gr.Column(scale=2):
    #             #     chx_prob = gr.Checkbox(value=True, label="Show probability")
    #             # with gr.Column(scale=2):
    #                 slider_top_k = gr.Slider(0, 5, step=1, interactive=True, label='Top K Result', value=3)
    #         with gr.Row():
    #             with gr.Column(scale=2):
    #             # text_answer = gr.Textbox(label="Label", interactive=False, placeholder="", lines=1, visible=True)
    #                 gr.Markdown("### Prediction of PROD_N_SERVICES")
    #                 text_pred_prod = gr.DataFrame(headers=["Label", "Probability"], max_rows=slider_top_k.value, interactive=False)
    #             with gr.Column(scale=2):
    #                 gr.Markdown("### Prediction of PROCESS_ISSUES")
    #                 text_pred_process = gr.DataFrame(headers=["Label", "Probability"], max_rows=slider_top_k.value, interactive=False)


    # slider_top_k.input(func_top_k, inputs=[slider_top_k], outputs=[text_pred_prod, text_pred_process])
        
    # btn_clean.click(func_clean, inputs=[text_input], outputs=[text_input])

    # btn_transcribe.click(func_prediction, inputs=[text_input, slider_top_k], outputs=[text_pred_prod, text_pred_process])
    # text_input.submit(func_prediction, inputs=[text_input, slider_top_k], outputs=[text_pred_prod, text_pred_process])
    # text_input.input(func_prediction_batch, inputs=[text_input, slider_top_k], outputs=[text_pred_prod, text_pred_process], batch=True, max_batch_size=8)

gr_app = gr_app.queue(concurrency_count=1) # concurrency_count=2)


if __name__ == "__main__":
    gr_app.launch(ssl_verify=False)
