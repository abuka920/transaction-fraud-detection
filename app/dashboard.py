"""Streamlit analyst review dashboard. Run: streamlit run app/dashboard.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import json

from src.analyst_queue import AnalystQueue, PENDING, CONFIRMED, FALSE_POS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ALERTS_PATH = PROJECT_ROOT / "alerts.json"

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🔍", layout="wide")


def bootstrap():
    """First run (e.g. a fresh clone or the hosted demo): train the model and
    fill the analyst queue so the dashboard has something to show."""
    import subprocess
    if (MODELS_DIR / "ensemble_model.joblib").exists() and ALERTS_PATH.exists():
        return
    with st.spinner("First run: generating data, training the model and scoring alerts (about 1-2 minutes)..."):
        run = lambda *args: subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT,
                                           check=True, capture_output=True, text=True)
        if not (MODELS_DIR / "ensemble_model.joblib").exists():
            run("train.py")
        threshold = 0.4
        metrics_path = MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            threshold = round(json.loads(metrics_path.read_text()).get("threshold", 0.4), 2)
        run("predict.py", "--input", "sample_data/sample_transactions.csv",
            "--threshold", str(threshold), "--max-alerts", "50")


bootstrap()


def get_queue():
    if "queue" not in st.session_state:
        st.session_state["queue"] = AnalystQueue(persist_path=ALERTS_PATH)
    return st.session_state["queue"]


def main():
    queue = get_queue()
    stats = queue.get_queue_stats()

    st.sidebar.title("🏦 Fraud Operations")
    st.sidebar.metric("🔴 Pending", stats["pending"])
    st.sidebar.metric("✅ Confirmed Fraud", stats["confirmed_fraud"])
    st.sidebar.metric("❌ False Positives", stats["false_positive"])
    st.sidebar.metric("💸 Pending Est. Loss", f"${stats['pending_estimated_loss']:,.2f}")
    if st.sidebar.button("🔄 Refresh"):
        st.session_state.pop("queue", None); st.rerun()

    st.title("🔍 Transaction Fraud Detection — Analyst Dashboard")

    tab1, tab2, tab3 = st.tabs(["🚨 Alert Queue", "📂 Review History", "📊 Model Metrics"])

    with tab1:
        pending = queue.get_all_alerts(status=PENDING)
        if not pending:
            st.success("No pending alerts — queue is clear!")
        else:
            st.subheader(f"📋 {len(pending)} Pending Alerts (sorted by estimated loss)")
            df = pd.DataFrame([{
                "Transaction ID": a.transaction_id, "Amount": f"${a.amount:,.2f}",
                "Fraud Prob": f"{a.fraud_probability:.1%}", "Est. Loss": f"${a.estimated_loss:,.2f}",
                "Top Reason": a.top_reason, "Timestamp": a.timestamp, "_id": a.alert_id,
            } for a in pending])
            st.dataframe(df.drop(columns=["_id"]), width="stretch", height=280)

            st.subheader("🔎 Review Alert")
            opts = {f"{a.transaction_id} — ${a.estimated_loss:,.2f}": a.alert_id for a in pending}
            sel_label = st.selectbox("Select:", list(opts.keys()))
            sel_id = opts[sel_label]
            sel = next((a for a in pending if a.alert_id == sel_id), None)
            if sel:
                c1, c2, c3 = st.columns(3)
                c1.metric("Amount", f"${sel.amount:,.2f}")
                c2.metric("Fraud Probability", f"{sel.fraud_probability:.1%}")
                c3.metric("Est. Loss", f"${sel.estimated_loss:,.2f}")
                st.markdown(f"**SHAP Explanation:** {sel.explanation}")
                note = st.text_input("Analyst note:", key=f"note_{sel_id}")
                b1, b2 = st.columns(2)
                if b1.button("🚨 Confirm Fraud", type="primary"):
                    queue.update_status(sel_id, CONFIRMED, note)
                    st.session_state.pop("queue", None); st.success("Confirmed fraud"); st.rerun()
                if b2.button("✅ False Positive"):
                    queue.update_status(sel_id, FALSE_POS, note)
                    st.session_state.pop("queue", None); st.info("Marked false positive"); st.rerun()

    with tab2:
        reviewed = queue.get_all_alerts(status=CONFIRMED) + queue.get_all_alerts(status=FALSE_POS)
        if not reviewed:
            st.info("No reviewed alerts yet.")
        else:
            st.dataframe(pd.DataFrame([{
                "Transaction ID": a.transaction_id, "Amount": f"${a.amount:,.2f}",
                "Status": a.status, "Reviewed At": a.reviewed_at or "",
                "Analyst Note": a.analyst_note or "",
            } for a in sorted(reviewed, key=lambda x: x.reviewed_at or "", reverse=True)]),
            width="stretch")

    with tab3:
        metrics_path = MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            c1, c2, c3 = st.columns(3)
            c1.metric("AUC-PR", f"{m.get('auc_pr',0):.4f}")
            c2.metric("F1 Score", f"{m.get('f1',0):.4f}")
            c3.metric("Threshold", f"{m.get('threshold',0.5):.2f}")
        else:
            st.info("Run `python train.py` to generate model metrics.")

        for img, cap in [(MODELS_DIR/"pr_curve.png","PR Curve"),
                          (MODELS_DIR/"confusion_matrix.png","Confusion Matrix")]:
            if img.exists():
                st.image(str(img), caption=cap, width="stretch")

        st.markdown("""
### Why AUC-PR > Accuracy for Fraud

With ~0.5% fraud rate, a model predicting "always legit" gets 99.5% accuracy — and catches zero fraud.

**Precision-Recall AUC** captures the tradeoff analysts care about:
- **Recall** = what fraction of real fraud did we catch (financial exposure)
- **Precision** = what fraction of flagged transactions are actually fraud (analyst workload)

Lowering the threshold catches more fraud but creates more false alarms. This dashboard lets analysts tune that tradeoff in real-time.
        """)


if __name__ == "__main__":
    main()
