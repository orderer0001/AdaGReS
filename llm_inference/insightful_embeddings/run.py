from argparse import ArgumentParser

import uvicorn

from insightful_embeddings.app import init_model, app

def run_embeddings(model_path,host,port,model_name="insightful_embeddings_model"):
    init_model(model_name,model_path)
    uvicorn.run(app, host=host, port=port)
