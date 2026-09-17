"""SHAP explanations for LightGBM model-attributed influence."""

import pandas as pd
import shap
from lightgbm import LGBMRegressor

from ..forecasting.models import EXCLUDED_FEATURES


def explain_lightgbm(train: pd.DataFrame, target: str = "target") -> pd.DataFrame:
    feature_columns = [c for c in train.columns if c not in EXCLUDED_FEATURES]
    clean = train.dropna(subset=feature_columns + [target])
    model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=7, verbosity=-1)
    model.fit(clean[feature_columns], clean[target])
    values = shap.TreeExplainer(model)(clean[feature_columns]).values
    return pd.DataFrame({"feature_name": feature_columns, "shap_value": values.mean(axis=0)}).sort_values("shap_value", key=abs, ascending=False)
