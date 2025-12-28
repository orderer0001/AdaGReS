
from fastapi import FastAPI, HTTPException
import torch

from insightful_embeddings.embeddings import EmbeddingModel
from insightful_embeddings.schemas import EmbeddingResponse, EmbeddingRequest, EmbeddingData, UsageInfo, LoadModelRequest, LoadModelResponse

app = FastAPI()

models: dict[str, EmbeddingModel] = {}


# 初始化模型
def init_model(model_name, model_path):
    print(f"加载模型 {model_path}...")
    model_name = model_name.lower()
    if model_name in models:
        print(f"模型 {model_path} 已经加载。")
    else:
        try:
            models[model_name] = EmbeddingModel(model_name, model_path)
            print(f"加载模型 {model_path} 成功。")
        except Exception as e:
            print(f"加载模型 {model_path} 失败：{str(e)}")


@app.post("/load", response_model=LoadModelResponse, response_model_exclude_none=True)
async def load_model(request: LoadModelRequest):
    try:
        if request.model in models:
            raise HTTPException(status_code=400, detail="Model already loaded")
        if request.path is None:
            raise HTTPException(status_code=400, detail="Path is required")
        init_model(request.model, request.path)
        return LoadModelResponse(model=request.model, loaded=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/unload", response_model=LoadModelResponse, response_model_exclude_none=True)
async def unload_model(request: LoadModelRequest):
    try:
        model_name = request.model.lower()
        if model_name not in models:
            raise HTTPException(status_code=404, detail="Model not found")

        models[model_name].unload()
        del models[model_name]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return LoadModelResponse(model=request.model, loaded=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/v1/embeddings", response_model=EmbeddingResponse, response_model_exclude_none=True)
async def embeddings(request: EmbeddingRequest):
    try:
        embedding_model = models.get(request.model.lower())
        if embedding_model is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Process input
        inputs = [request.input] if isinstance(request.input, str) else request.input
        # Calculate token count
        input_token_counts = embedding_model.calculate_token_count(inputs)
        total_tokens_used = sum(input_token_counts)
        # Generate embeddings
        embeddings = embedding_model.generate_embeddings(inputs, encoding_format=request.encoding_format)
        # Build response
        response = EmbeddingResponse(
            model=request.model,
            object="list",
            data=[EmbeddingData(embedding=emb, index=idx, object="embedding") for idx, emb in enumerate(embeddings)],
            user=request.user,
            usage=UsageInfo(
                prompt_tokens=total_tokens_used,
                total_tokens=total_tokens_used
            )
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


