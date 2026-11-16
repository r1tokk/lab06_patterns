import pandas as pd
from src.local import LocalModelRepository

class PredictionService:
    def __init__(self, repository: LocalModelRepository, version="v1"):
        self._pipeline, self._schema, self._meta = repository.load(version)
        self.version = version

    def predict_one(self, payload: dict) -> float:
        frame = pd.DataFrame([payload])[self._schema["feature_order"]]

        return float(self._pipeline.predict(frame)[0])
