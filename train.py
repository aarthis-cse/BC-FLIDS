"""
BC-FLIDS: Blockchain-Federated Learning Intrusion Detection System
train.py — Training pipeline (faithful to original bc_flids_model.py)

This script reproduces the exact training procedure used to generate
the paper's reported results, with the following corrections:
  - Holdout split is done BEFORE scaling (original already did this correctly)
  - Scaler is fitted on train only (same as original)
  - No feature selection is applied (original used all numeric features)
  - Loss: weighted CrossEntropyLoss with sklearn class weights (same as original)
  - Architecture: 2-layer BiLSTM + custom attention (same as original)

Usage:
  python train.py --data IoV_SecFusion_Dataset.csv --output_dir ./outputs
"""

import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight

from model import BC_FLIDS

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# CONFIG — mirrors original bc_flids_model.py exactly
# ──────────────────────────────────────────────────────────────
CONFIG = {
    "epochs"              : 50,
    "batch_size"          : 32,
    "learning_rate"       : 0.001,
    "weight_decay"        : 1e-4,
    "hidden_size"         : 128,
    "num_layers"          : 2,
    "dropout"             : 0.3,
    "n_folds"             : 5,
    "test_size"           : 0.20,
    "early_stop_patience" : 10,
    "seed"                : 42,
    "label_col"           : "Label",
    "source_col"          : "dataset_source",
}

SEED = CONFIG["seed"]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ──────────────────────────────────────────────────────────────
# Data loading (same logic as original)
# ──────────────────────────────────────────────────────────────
def load_data(csv_path: str):
    print(f"\n{'='*55}")
    print("STEP 1: Loading merged dataset...")
    print(f"{'='*55}")

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Loaded: {df.shape[0]:,} rows | {df.shape[1]} columns")

    label_col = CONFIG["label_col"]
    source_col = CONFIG["source_col"]

    # Drop non-feature columns
    df.drop(columns=[c for c in [source_col] if c in df.columns], inplace=True)

    # Drop rows with missing label
    df.dropna(subset=[label_col], inplace=True)

    # Encode label
    df[label_col] = df[label_col].astype(str).str.strip()
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    num_classes = len(le.classes_)
    print(f"  Classes ({num_classes}): {list(le.classes_)}")

    # Features
    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])

    # Fill NaN → median, then replace inf → 0
    X.fillna(X.median(), inplace=True)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    print(f"  Feature count: {X.shape[1]}")
    return X.values.astype(np.float32), y, le


# ──────────────────────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────────────────────
def main(args):
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── STEP 1: Load data ────────────────────────────────────
    X_all, y_all, le = load_data(args.data)
    num_classes = len(le.classes_)
    input_size  = X_all.shape[1]

    # ── STEP 2: Holdout split (20%) ──────────────────────────
    print(f"\n{'='*55}")
    print("STEP 2: Creating holdout test set (20%)...")
    print(f"{'='*55}")
    X_model, X_test, y_model, y_test = train_test_split(
        X_all, y_all,
        test_size=CONFIG["test_size"],
        random_state=SEED,
        stratify=y_all
    )
    print(f"  Model data (80%): {X_model.shape[0]:,} rows")
    print(f"  Holdout test (20%): {X_test.shape[0]:,} rows")

    # ── STEP 3: Scale features (fit on train only) ───────────
    scaler  = StandardScaler()
    X_model = scaler.fit_transform(X_model)
    X_test  = scaler.transform(X_test)

    # Reshape for LSTM: (N, 1, features)
    X_model_t = torch.FloatTensor(X_model).unsqueeze(1)
    X_test_t  = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_t  = torch.LongTensor(y_test)

    # ── STEP 4: 5-Fold CV ────────────────────────────────────
    print(f"\n{'='*55}")
    print("STEP 5: Running 5-Fold Cross Validation...")
    print(f"{'='*55}")

    skf = StratifiedKFold(n_splits=CONFIG["n_folds"],
                          shuffle=True, random_state=SEED)
    fold_results    = []
    best_model_state = None
    best_val_f1      = 0.0

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(X_model_t, y_model)):
        print(f"\n  ── Fold {fold+1}/{CONFIG['n_folds']} ──")

        X_tr = X_model_t[train_idx].to(device)
        y_tr = torch.LongTensor(y_model[train_idx]).to(device)
        X_vl = X_model_t[val_idx].to(device)
        y_vl = torch.LongTensor(y_model[val_idx]).to(device)

        train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=CONFIG["batch_size"], shuffle=True)
        val_dl   = DataLoader(TensorDataset(X_vl, y_vl),
                              batch_size=CONFIG["batch_size"], shuffle=False)

        # Fresh model per fold
        model = BC_FLIDS(
            input_size=input_size,
            hidden_size=CONFIG["hidden_size"],
            num_layers=CONFIG["num_layers"],
            num_classes=num_classes,
            dropout=CONFIG["dropout"],
        ).to(device)

        # Class weights for imbalance handling (same as original)
        cw = compute_class_weight(
            'balanced',
            classes=np.unique(y_model[train_idx]),
            y=y_model[train_idx]
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(cw).to(device))
        optimizer = optim.AdamW(model.parameters(),
                                lr=CONFIG["learning_rate"],
                                weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False)

        best_val_loss   = float('inf')
        patience_count  = 0
        best_fold_state = None

        for epoch in range(CONFIG["epochs"]):
            # Train
            model.train()
            train_loss = 0.0
            for X_b, y_b in train_dl:
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dl)

            # Validate
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_b, y_b in val_dl:
                    out = model(X_b)
                    val_loss += criterion(out, y_b).item()
                    all_preds.extend(torch.argmax(out, 1).cpu().numpy())
                    all_labels.extend(y_b.cpu().numpy())
            val_loss /= len(val_dl)
            scheduler.step(val_loss)

            val_acc = accuracy_score(all_labels, all_preds)
            val_f1  = f1_score(all_labels, all_preds,
                               average='weighted', zero_division=0)
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d} | Train: {train_loss:.4f} "
                      f"| Val: {val_loss:.4f} | Acc: {val_acc:.4f} "
                      f"| F1: {val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                patience_count  = 0
                best_fold_state = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= CONFIG["early_stop_patience"]:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        # Fold evaluation
        model.load_state_dict(best_fold_state)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                all_preds.extend(
                    torch.argmax(model(X_b), 1).cpu().numpy())
                all_labels.extend(y_b.cpu().numpy())

        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds,
                               average='weighted', zero_division=0)
        rec  = recall_score(all_labels, all_preds,
                            average='weighted', zero_division=0)
        f1   = f1_score(all_labels, all_preds,
                        average='weighted', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_preds,
                                multi_class='ovr', average='weighted')
        except Exception:
            auc = 0.0

        fold_results.append({"fold": fold+1, "acc": acc, "prec": prec,
                              "rec": rec, "f1": f1, "auc": auc})
        print(f"  Fold {fold+1} → Acc:{acc:.4f} Prec:{prec:.4f} "
              f"Rec:{rec:.4f} F1:{f1:.4f} AUC:{auc:.4f}")

        if f1 > best_val_f1:
            best_val_f1      = f1
            best_model_state = best_fold_state

    # CV summary
    print(f"\n{'='*55}")
    print("STEP 6: Cross Validation Summary")
    print(f"{'='*55}")
    results_df = pd.DataFrame(fold_results)
    print(f"\n  {'Metric':<12} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*30}")
    for m in ["acc", "prec", "rec", "f1", "auc"]:
        mean_val = results_df[m].mean()
        std_val  = results_df[m].std()
        print(f"  {m.upper():<12} {mean_val:>8.4f} {std_val:>8.4f}")

    results_df.to_csv(
        os.path.join(args.output_dir, 'v3_cv_results.csv'), index=False)

    # ── STEP 7: Final holdout evaluation ─────────────────────
    print(f"\n{'='*55}")
    print("STEP 7: Final evaluation on holdout test set...")
    print(f"{'='*55}")

    final_model = BC_FLIDS(
        input_size=input_size,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_classes=num_classes,
        dropout=CONFIG["dropout"],
    ).to(device)
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    test_dl = DataLoader(
        TensorDataset(X_test_t.to(device), y_test_t.to(device)),
        batch_size=CONFIG["batch_size"], shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_b, y_b in test_dl:
            out   = final_model(X_b)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds,
                           average='weighted', zero_division=0)
    rec  = recall_score(all_labels, all_preds,
                        average='weighted', zero_division=0)
    f1   = f1_score(all_labels, all_preds,
                    average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, np.array(all_probs),
                            multi_class='ovr', average='weighted')
    except Exception:
        auc = 0.0

    print(f"\n  FINAL TEST RESULTS (Holdout 20%)")
    print(f"  {'─'*38}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}  ({prec*100:.2f}%)")
    print(f"  Recall    : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  AUC-ROC   : {auc:.4f}  ({auc*100:.2f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=le.classes_, zero_division=0))
    print(f"  Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")

    # ── STEP 8: Save ─────────────────────────────────────────
    import joblib
    torch.save(best_model_state,
               os.path.join(args.output_dir, 'bc_flids_v3.pth'))
    joblib.dump(scaler,
                os.path.join(args.output_dir, 'v3_scaler.pkl'))
    joblib.dump(le,
                os.path.join(args.output_dir, 'v3_label_encoder.pkl'))
    np.save(os.path.join(args.output_dir, 'v3_label_classes.npy'),
            le.classes_)

    pd.DataFrame([{"Accuracy": acc, "Precision": prec,
                   "Recall": rec, "F1_Score": f1, "AUC_ROC": auc}
                  ]).to_csv(
        os.path.join(args.output_dir, 'v3_test_results.csv'), index=False)
    np.save(os.path.join(args.output_dir, 'v3_test_preds.npy'),
            np.array(all_preds))
    np.save(os.path.join(args.output_dir, 'v3_test_labels.npy'),
            np.array(all_labels))
    np.save(os.path.join(args.output_dir, 'v3_test_probs.npy'),
            np.array(all_probs))

    print(f"\n  All outputs saved to: {args.output_dir}")
    print(f"{'='*55}")
    print("ALL DONE! BC-FLIDS training complete.")
    print(f"{'='*55}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BC-FLIDS model')
    parser.add_argument('--data',       default='IoV_SecFusion_Dataset.csv')
    parser.add_argument('--output_dir', default='./outputs')
    args = parser.parse_args()
    main(args)
