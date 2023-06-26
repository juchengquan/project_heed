from fastapi import FastAPI
from .endpoints import api_health, api_transribe_with_file

def get_app():
    app = FastAPI()
    
    app.add_api_route("/", api_health, methods=["GET", "POST"])
    app.add_api_route("/transcribe", api_transribe_with_file, methods=["POST"])
    
    return app

app = get_app()