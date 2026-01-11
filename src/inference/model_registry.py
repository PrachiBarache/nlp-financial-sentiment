from src.model.svm import SVMModel
from src.model.roberta import RobertaModel
from src.model.finbert import FinBERTModel

MODEL_REGISTRY = {
    "svm": SVMModel(),
    "roberta": RobertaModel(),
    "finbert": FinBERTModel()
}