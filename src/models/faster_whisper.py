from .model_config import MODEL_CONFIG
from ..logging import logger

from faster_whisper import WhisperModel

_model_config_whisper = MODEL_CONFIG["models"]["whisper"]

_path = _model_config_whisper["model_path"]
_device_index = _model_config_whisper["device"]
_model_name = _model_config_whisper["name"]

def get_MODEL_ASR():
    logger.info(f"Loading Whisper model...")
    return WhisperModel(
        model_size_or_path = _path,
        device_index = _device_index,
        # compute_type = "float16",
        device = "cpu", # TODO # torch.device(f"cuda:{_device}" if torch.cuda.is_available() else "cpu")
    )

logger.info(f"ASR generation config loaded.")
CONFIG_ASR = _model_config_whisper["generation_config"]
