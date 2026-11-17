import numpy as np
import shap

_LABELS = {
    "amount": "transaction amount", "hour_of_day": "time of day",
    "distance_from_home_km": "distance from home", "velocity_1h": "txns in last hour",
    "amount_z_score": "amount vs average", "merchant_risk_score": "merchant risk",
    "is_foreign_transaction": "foreign transaction", "card_present": "card not present",
    "is_night": "late-night hour", "amount_velocity": "high amount with velocity",
    "distance_foreign": "foreign + far from home",
}


def _label(f: str) -> str:
    for k, v in _LABELS.items():
        if f.startswith(k):
            return v
    return f.replace("cat_", "").replace("_", " ") + " merchant" if f.startswith("cat_") else f.replace("_"," ")


class FraudExplainer:
    def __init__(self, ensemble_model, feature_names: list):
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(ensemble_model.xgb_model,
                                             feature_perturbation="tree_path_dependent")

    def explain_batch(self, X: np.ndarray) -> list:
        sv = self.explainer.shap_values(X)
        results = []
        for i in range(len(X)):
            top = np.argsort(np.abs(sv[i]))[::-1][:5]
            contributions = [{"feature": self.feature_names[j], "label": _label(self.feature_names[j]),
                               "shap_value": round(float(sv[i][j]), 4),
                               "direction": "increases" if sv[i][j] > 0 else "reduces"} for j in top]
            results.append({"contributions": contributions})
        return results

    def explain_one(self, X_row: np.ndarray) -> dict:
        return self.explain_batch(X_row.reshape(1, -1))[0]

    def format_explanation(self, exp: dict) -> str:
        parts = [f"{c['label']} ({'+' if c['shap_value']>0 else ''}{c['shap_value']:.2f})"
                 for c in exp["contributions"]]
        return "Flagged because: " + ", ".join(parts)

    def top_reason(self, exp: dict) -> str:
        c = exp["contributions"][0]
        return f"{c['label']} ({'+' if c['shap_value']>0 else ''}{c['shap_value']:.2f})"
