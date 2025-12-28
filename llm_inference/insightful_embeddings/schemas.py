from pydantic import BaseModel
from typing import Union, List, Optional, Literal


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: Optional[int] = None
    total_tokens: int


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    dimensions: Optional[int] = None
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: Union[List[float], str]  # 支持 'float' 或 'base64' 格式
    index: int


class EmbeddingResponse(BaseModel):
    model: str
    object: str
    data: List[EmbeddingData]
    user: Optional[str] = None
    usage: UsageInfo


class LoadModelRequest(BaseModel):
    model: str
    path: Optional[str] = None


class LoadModelResponse(BaseModel):
    model: str
    loaded: bool


class HealthResponse(BaseModel):
    status: str
