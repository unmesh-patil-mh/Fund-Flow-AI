import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

# Add Fraud_Detection path to import modules
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fraud_detect_dir = os.path.join(base_dir, "Fraud_Detection", "Fraud_Detection")
sys.path.insert(0, fraud_detect_dir)

from features.engineering import engineer_features, get_feature_columns

MODEL_PATH = os.path.join(fraud_detect_dir, "models", "saved", "xgboost_fraud.pkl")
META_PATH = os.path.join(fraud_detect_dir, "models", "saved", "model_metadata.json")

def load_ml_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta

def map_prisma_transactions(txns_list):
    mapped_list = []
    for t in txns_list:
        sender = t.get("senderAccount", {}) or {}
        receiver = t.get("receiverAccount", {}) or {}
        
        # Parse timestamp safely
        ts = t.get("timestamp")
        if isinstance(ts, str):
            ts = ts.replace("Z", "").split(".")[0]
            try:
                dt = datetime.fromisoformat(ts)
            except:
                dt = datetime.now()
        else:
            dt = datetime.now()

        # Map transaction types
        type_map = {
            "CASH_DEPOSIT": "DEPOSIT",
            "CASH_WITHDRAWAL": "ATM",
            "TRANSFER": "NEFT",
            "PAYMENT": "UPI",
            "DEBIT": "IMPS",
        }
        txn_type = type_map.get(t.get("type"), t.get("type", "UPI"))

        # Map channels
        channel_map = {
            "MOBILE_APP": "mobile",
            "NET_BANKING": "internet",
            "ATM": "atm",
            "BRANCH": "branch",
            "POS": "pos",
            "API": "api",
        }
        channel = channel_map.get(t.get("channel"), "mobile")

        mapped = {
            "txn_id": t.get("transactionId", t.get("id", "")),
            "timestamp": dt,
            "sender_account": sender.get("accountNumber", "unknown_sender"),
            "receiver_account": receiver.get("accountNumber", "unknown_receiver"),
            "amount": float(t.get("amount", 0.0)),
            "txn_type": txn_type,
            "channel": channel,
            "sender_branch": "BR_MUMBAI_001",
            "receiver_branch": "BR_DELHI_001",
            "step": 1,
            "sender_balance_before": 0.0,
            "sender_balance_after": 0.0,
            "sender_mule_score": float(sender.get("muleScore", 0.0)),
            "receiver_mule_score": float(receiver.get("muleScore", 0.0)),
            "kyc_risk_flag": 1 if sender.get("kycType") == "MIN_KYC" else 0,
            "cibil_high_txn_flag": 1 if (sender.get("creditScore") and sender.get("creditScore") < 550) else 0,
            "is_fraud": 1 if t.get("isFraud") is True or t.get("isFraud") == 1 else 0
        }
        mapped_list.append(mapped)
    return pd.DataFrame(mapped_list)

def run_predictions_on_df(df, model, meta):
    feature_cols = meta["feature_columns"]
    
    # We will try to engineer features on this df
    df_feat = engineer_features(df)
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
            
    X = df_feat[feature_cols].fillna(0.0)
    probs = model.predict_proba(X)[:, 1]
    
    # Platt scaling calibration parameters
    A, B = -92.0, 88.0
    calibrated = 1.0 / (1.0 + np.exp(A * probs + B))
    calibrated = np.clip(calibrated, 0.0, 1.0)
    
    df = df.copy()
    df["calibrated_prob"] = calibrated
    df["predicted_label"] = (calibrated >= 0.5).astype(int)
    return df

def cmd_evaluate(txns_list):
    model, meta = load_ml_model()
    if not model or not meta:
        print(json.dumps({"error": "Model not trained"}))
        return

    if len(txns_list) < 5:
        print(json.dumps({"error": "Insufficient Data"}))
        return

    df = map_prisma_transactions(txns_list)
    df_pred = run_predictions_on_df(df, model, meta)
    
    y_true = df_pred["is_fraud"].values
    y_prob = df_pred["calibrated_prob"].values
    y_pred = df_pred["predicted_label"].values
    
    # Compute classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle single class case for AUC-ROC
    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_prob)
    else:
        auc_roc = 0.5
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    if len(cm) == 1:
        if y_true[0] == 0:
            cm = [[cm[0][0], 0], [0, 0]]
        else:
            cm = [[0, 0], [0, cm[0][0]]]
            
    fi = meta.get("feature_importance", {})
    
    result = {
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc_roc),
            "accuracy": float(accuracy),
            "confusion_matrix": cm,
            "auc_pr": float(meta.get("metrics", {}).get("auc_pr", 0.5))
        },
        "feature_importance": fi,
        "modelName": meta.get("model_type", "XGBClassifier"),
        "version": "v1",
        "type": "XGBoost",
        "description": "Dynamic evaluation on all Neon database transactions",
        "isMLActive": True,
        "features": meta.get("feature_columns", [])
    }
    print(json.dumps(result))

def cmd_predict(txn_data):
    model, meta = load_ml_model()
    if not model or not meta:
        print(json.dumps({"error": "Model not trained"}))
        return
    
    df = map_prisma_transactions([txn_data])
    df_pred = run_predictions_on_df(df, model, meta)
    
    prob = float(df_pred["calibrated_prob"].iloc[0])
    label = int(df_pred["predicted_label"].iloc[0])
    
    # Predict confidence
    confidence = round(2.0 * abs(prob - 0.5), 4)
    
    # Get top contributing features
    fi = meta.get("feature_importance", {})
    top_features = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    result = {
        "fraudScore": round(prob, 4),
        "isFraud": label == 1,
        "confidence": confidence,
        "reasons": [
            {
                "feature": k,
                "impact": round(float(v), 4),
                "value": round(float(v), 4),
                "description": k.replace("_", " ").title()
            } for k, v in top_features
        ],
        "modelVersion": "xgboost-v1"
    }
    print(json.dumps(result))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing action argument"}))
        sys.exit(1)
        
    action = sys.argv[1]
    input_data = json.loads(sys.stdin.read())
    
    if action == "--eval":
        cmd_evaluate(input_data)
    elif action == "--predict":
        cmd_predict(input_data)
    else:
        print(json.dumps({"error": "Unknown action"}))
        sys.exit(1)
