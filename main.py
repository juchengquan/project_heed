import os, argparse
import uvicorn
import pathlib
from src.logging import logger

import torch
if torch.cuda.device_count() > 0:
    print(f"__CUDA Device Name: {torch.cuda.get_device_name(0)}")


os.environ["PROJ_DIR"] = str(pathlib.Path(__file__).parent.resolve())
print(os.environ["PROJ_DIR"])

def main():
    _service_port = os.environ.get("SERVICE_PORT", 8080)
    logger.info(f"SERVICE_PORT: {_service_port}")
    
    config = uvicorn.Config("src.server:app",
                            host="localhost",
                            port=int(_service_port),
                            log_level="debug",
                            access_log=False)
    server = uvicorn.Server(config)
    return server
    # await server.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--MODEL_CONFIG_FILE", type=str, required=True, help="MODEL_CONFIG_FILE")
    parser.add_argument("-p", "--SERVICE_PORT", type=str, default="8080", help="SERVICE_PORT")
    
    args = parser.parse_args()
    os.environ["SERVICE_PORT"] = args.SERVICE_PORT
    os.environ["MODEL_CONFIG_FILE"] = args.MODEL_CONFIG_FILE
    
    server = main()
    server.run()
    # asyncio.run(main())
