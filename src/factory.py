from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

def get_model(name: str, random_state: int = 42):
    if name == "ridge":
        return Ridge(alpha=10.0)
    elif name == "forest":
        return RandomForestRegressor(n_estimators=200, random_state=random_state)
    return DummyRegressor(strategy="median")