import joblib
import json
from pathlib import Path

class LocalModelRepository:
    def __init__(self, base_dir="artifacts"):
        self.base_dir = Path(base_dir)

    def save(self, pipeline, schema, metadata, version="v1"):
        path = self.base_dir / version
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path / "model.joblib") #[cite: 7]
        with open(path / "feature_schema.json", "w") as f: json.dump(schema, f) #[cite: 7]
        with open(path / "metadata.json", "w") as f: json.dump(metadata, f) #[cite: 7]

    def load(self, version="v1"):
        path = self.base_dir / version
        pipeline = joblib.load(path / "model.joblib")
        with open(path / "feature_schema.json", "r") as f: schema = json.load(f)
        with open(path / "metadata.json", "r") as f: metadata = json.load(f)
        return pipeline, schema, metadata