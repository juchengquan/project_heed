import torch
from .pretrained import SpeechBrainPretrainedSpeakerEmbedding

from .model_config import MODEL_CONFIG
from ..logging import logger

path_embedding: str = MODEL_CONFIG["models"]["embedding"]["model_path"]
_device = MODEL_CONFIG["models"]["embedding"]["device"]

def get_MODEL_EMBEDDING():
    logger.info("Loading embedding model...")
    return SpeechBrainPretrainedSpeakerEmbedding(
        embedding = path_embedding,
        device = torch.device(f"cuda:{_device}" if torch.cuda.is_available() else "cpu")
    )
    
