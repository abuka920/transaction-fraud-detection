"""Score transactions and populate analyst queue. Usage: python predict.py --input FILE"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import click, numpy as np, pandas as pd
from src.model import load_model
from src.preprocess import preprocess_new
from src.explain import FraudExplainer
from src.analyst_queue import AnalystQueue

MODELS_DIR = Path("models")
ALERTS_PATH = Path("alerts.json")


@click.command()
@click.option("--input", "inp", required=True, type=click.Path(exists=True))
@click.option("--threshold", default=0.3, show_default=True)
@click.option("--output", default="alerts_output.csv")
@click.option("--max-alerts", default=None, type=int)
def predict(inp, threshold, output, max_alerts):
    """Score transactions and push alerts to analyst queue."""
    print("Loading model artifacts...")
    try:
        model, scaler, feature_names = load_model(MODELS_DIR)
    except FileNotFoundError:
        click.echo("Run python train.py first.", err=True); sys.exit(1)
    model.threshold_ = threshold

    df = pd.read_csv(inp)
    print(f"Loaded {len(df):,} transactions")

    X = preprocess_new(df, scaler, feature_names)
    probs = model.predict_proba(X)[:,1]
    df = df.copy()
    df["fraud_probability"] = np.round(probs, 4)
    df["is_alert"] = (probs >= threshold).astype(int)

    n_alerts = int(df["is_alert"].sum())
    print(f"Flagged {n_alerts:,} at threshold {threshold:.2f} ({n_alerts/len(df)*100:.2f}%)")
    df.to_csv(output, index=False)

    print("Generating SHAP explanations...")
    explainer = FraudExplainer(model, feature_names)
    alert_df = df[df["is_alert"]==1].copy()
    if max_alerts:
        alert_df = alert_df.nlargest(max_alerts, "fraud_probability")
    # Select feature rows by the same index so each explanation matches its transaction
    X_alerts = X[alert_df.index.to_numpy()]

    explanations = explainer.explain_batch(X_alerts)
    queue = AnalystQueue(persist_path=ALERTS_PATH)
    for row, exp in zip(alert_df.to_dict("records"), explanations):
        queue.add_alert(
            transaction_id=str(row.get("transaction_id","?")),
            amount=float(row.get("amount",0)),
            fraud_probability=float(row["fraud_probability"]),
            top_reason=explainer.top_reason(exp),
            explanation=explainer.format_explanation(exp),
            timestamp=str(row.get("timestamp","")),
        )

    stats = queue.get_queue_stats()
    print(f"Queue: {stats}")
    print(f"Run: streamlit run app/dashboard.py")


if __name__ == "__main__":
    predict()
