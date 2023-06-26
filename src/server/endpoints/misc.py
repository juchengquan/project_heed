from io import BytesIO
import os
import hashlib, uuid
from datetime import datetime

def get_trace_id(file_name: str = None):
    if isinstance(file_name, str) and file_name:
        _name = (file_name + "_" + str(datetime.now().timestamp())).encode("utf-8")
    else:
        _name = str(datetime.now().timestamp()).encode("utf-8")
    
    return hashlib.sha256(_name).hexdigest()

