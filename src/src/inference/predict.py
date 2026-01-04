import time
from src.inference.model_registry import MODEL_REGISTRY

def predict(text: str, model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError("Unsupported model")

    model = MODEL_REGISTRY[model_name]
    start = time.time()
    sentiment, confidence = model.predict(text)
    latency = int((time.time() - start) * 1000)

    return {
        "model_used": model_name,
        "sentiment": sentiment,
        "confidence": confidence,
        "latency_ms": latency
    }
