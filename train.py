"""Train the fraud detection ensemble. Usage: python train.py"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_generator import generate_transactions
from src.preprocess import preprocess
from src.model import FraudEnsemble, evaluate, save_model

MODELS_DIR = Path("models")
DATA_PATH = Path("sample_data/sample_transactions.csv")


def main():
    print("=" * 55)
    print("  Transaction Fraud Detection — Training Pipeline")
    print("=" * 55)

    if not DATA_PATH.exists():
        print("\n[1/4] Generating synthetic transactions...")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = generate_transactions(n_total=100_000, fraud_rate=0.005)
        df.to_csv(DATA_PATH, index=False)
    else:
        print(f"\n[1/4] Using existing data: {DATA_PATH}")

    print("\n[2/4] Preprocessing + SMOTE...")
    data = preprocess(DATA_PATH)

    print("\n[3/4] Training XGBoost + LightGBM ensemble...")
    model = FraudEnsemble(random_state=42)
    model.fit(data["X_train"], data["y_train"], feature_names=data["feature_names"])
    threshold = model.tune_threshold(data["X_val"], data["y_val"])
    print(f"Alert threshold picked on validation set: {threshold:.2f}")

    print("\n[4/4] Evaluating on held-out test set (never seen in training or tuning)...")
    metrics = evaluate(model, data["X_test"], data["y_test"], out_dir=MODELS_DIR)
    save_model(model, data["scaler"], data["feature_names"], MODELS_DIR)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n" + "=" * 55)
    print(f"  Done. AUC-PR: {metrics['auc_pr']:.4f} | F1: {metrics['f1']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
