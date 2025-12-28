from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import base64
import torch
import gc


class EmbeddingModel:
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = SentenceTransformer(model_path)

    def calculate_token_count(self, inputs):
        encoded_inputs = self.tokenizer.batch_encode_plus(inputs, add_special_tokens=False)
        return [len(tokens) for tokens in encoded_inputs['input_ids']]

    def generate_embeddings(self, inputs, encoding_format="float"):
        embeddings = self.model.encode(inputs, batch_size=16)

        if encoding_format == "base64":
            embeddings = [base64.b64encode(emb.numpy().tobytes()).decode('utf-8') for emb in embeddings]
        elif encoding_format == "float":
            embeddings = embeddings.tolist()
        else:
            raise ValueError("Invalid encoding_format. Choose 'float' or 'base64'.")

        return embeddings

    def get_model_name(self):
        return self.model_name

    def unload(self):
        self.model.to("cpu")
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
