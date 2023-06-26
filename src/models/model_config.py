import yaml

import os
from ..logging import logger

PROJ_DIR = os.environ["PROJ_DIR"]

if not os.environ["MODEL_CONFIG_FILE"]:
    file_name = f"{PROJ_DIR}/configs/model_config.yaml"
else:
    file_name = os.environ["MODEL_CONFIG_FILE"]

with open(file_name, "r") as f:
    MODEL_CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(f"Read model configuration: {file_name}")
    