from fastapi import FastAPI
from .endpoints import api_health, api_transribe_with_file

# from .model_binding import get_MODEL_EMBEDDING, get_MODEL_ASR, \
#     CONFIG_ASR, standardize_audio_format, get_transcription_segments, get_duration, get_diarization

# MODEL_EMBEDDING = get_MODEL_EMBEDDING()
# MODEL_ASR = get_MODEL_ASR()

def get_app():
    app = FastAPI()
    
    app.add_api_route("/", api_health, methods=["GET", "POST"])
    app.add_api_route("/transcribe", api_transribe_with_file, methods=["POST"])
    
    return app

app = get_app()