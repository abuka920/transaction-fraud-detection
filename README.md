# Transaction Fraud Detection System

XGBoost + LightGBM ensemble for card fraud detection on heavily imbalanced data (0.5% fraud), with SHAP explanations for every alert and an analyst review queue. Built with my fraud investigation work supporting CIBC in mind.

## The core problem

**Catching more fraud means more false alarms.** At a 0.5% fraud rate, a model that calls everything "legit" scores 99.5% accuracy and catches nothing. So this project is measured on precision (how many alerts are real fraud) and recall (how much fraud gets caught), not accuracy.

## Results

Measured on a held-out test set of 20,000 transactions (100 fraud) the model never saw during training or tuning:

| Metric | Score |
|--------|-------|
| AUC-PR | ~0.63 |
| Precision at chosen threshold | ~0.80 (4 of 5 alerts are real fraud) |
| Recall at chosen threshold | ~0.55 (catches about half the fraud) |

For reference, random guessing gets an AUC-PR of 0.005 (the fraud rate).

Why not 99%? The synthetic data includes **stealthy fraud** (about a third of cases) that looks like normal spending: domestic, daytime, card present, modest amounts. Some **legitimate customers look risky** too, like travellers spending abroad. Real fraud works this way, and a near-perfect score would mean the data was too easy.

## Doing the evaluation honestly

The data is split **before** anything is fitted:

| Split | Size | Used for |
|-------|------|----------|
| Train | 60% | Fitting the scaler, SMOTE oversampling, training the models |
| Validation | 20% | Picking the alert threshold |
| Test | 20% | Final score, touched once |

Tuning the threshold on the test set, or fitting the scaler on all the data, leaks test information into the model and inflates the score. Both are avoided here.

## SHAP explanations for analysts

Every flagged transaction shows what drove the score:

```
Flagged because: distance from home (+2.61), txns in last hour (+2.27),
transaction amount (+1.48), foreign + far from home (+1.29), card not present (+1.02)
```

Analysts can see why something was flagged instead of trusting a black box. Explainability is also what model-risk guidance like OSFI's Guideline E-23 expects from banks.

## Threshold tuning

| Threshold | Effect |
|-----------|--------|
| Lower (0.2–0.3) | More alerts, higher recall, more analyst work |
| Validation-tuned (~0.4) | Best balance of precision and recall (F1) |
| Higher (0.6–0.7) | Fewer alerts, higher precision, more fraud missed |

## Quick start

```bash
pip install -r requirements.txt

# Train (generates synthetic data automatically)
python train.py

# Score transactions and fill the analyst queue
python predict.py --input sample_data/sample_transactions.csv --threshold 0.4 --max-alerts 50

# Launch the analyst dashboard
streamlit run app/dashboard.py
```

The dashboard trains the model and fills the queue by itself on first launch, so `streamlit run app/dashboard.py` on a fresh clone works too.

## Tech stack

- **XGBoost + LightGBM**: weighted ensemble
- **SMOTE**: oversamples fraud in the training split only
- **SHAP**: TreeExplainer on the XGBoost component
- **Streamlit**: analyst queue with Confirm Fraud / False Positive workflow
- **Click**: CLI for training and scoring
