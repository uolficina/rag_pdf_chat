from sentence_transformers import SentenceTransformer, CrossEncoder

from rag.config import (
    CROSSENCODER_MODEL,
    EMBEDDING_MODEL_NAME,
    state,
)


def load_models():
   
    if state["embed_model"] is None:
        state["embed_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("EMBEDDING MODEL LOADED")

    if state["cross_encoder"] is None:
        state["cross_encoder"] = CrossEncoder(CROSSENCODER_MODEL)
        print("CROSS-ENCODER LOADED")

    return state["embed_model"], state["cross_encoder"]


def get_models():
    
    if state["embed_model"] is None or state["cross_encoder"] is None:
        return load_models()
    return state["embed_model"], state["cross_encoder"]

