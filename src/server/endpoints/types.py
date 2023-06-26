from fastapi import Form, File, UploadFile, Depends, HTTPException
from pydantic import BaseModel

from typing import Optional

from .misc import get_trace_id

class BaseFilePayload(BaseModel):
    trace_id: Optional[str] = None
    language: Optional[str] = None
    if_diarize: Optional[str] = None
    file_name: Optional[UploadFile] = File(...)

def validate_file_payload(payload: BaseFilePayload):
    payload.language = "en" if not payload.language else payload.language
    payload.trace_id = get_trace_id()
    payload.if_diarize = str2bool(payload.if_diarize)
    return payload

def str2bool(v: str):
    if v:
        return v.lower() in ("true")
    return False
