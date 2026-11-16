from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class NoOpScaler(BaseEstimator, TransformerMixin):
    """Стратегія 'none': масштабування не виконується."""
    def fit(self, X, y=None): return self
    def transform(self, X, y=None): return X

def get_scaler(strategy: str):
    if strategy == "standard": return StandardScaler()
    elif strategy == "minmax": return MinMaxScaler()
    return NoOpScaler()