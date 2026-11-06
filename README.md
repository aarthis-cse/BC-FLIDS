# BC-FLIDS: Blockchain-Federated Learning Intrusion Detection System for IoV

> **Reproducible codebase** for the paper  
> *"BC-FLIDS: A Blockchain-Enhanced Federated Learning Framework for Intrusion Detection in Internet of Vehicles"*

---

## Repository structure

```
bc_flids_repo/
├── model.py          # 2-layer BiLSTM + custom additive attention
├── train.py          # Full training pipeline (5-fold CV + holdout)
├── ablations.py      # Ablation study (5 variants × 5 folds)
├── metrics.py        # Novel metrics: ARS, GSPP/PPI, CLR, DEM
├── requirements.txt  # Python dependencies
└── outputs/          # Saved models, CSVs (generated after running)
    ├── bc_flids_v3.pth
    ├── v3_cv_results.csv
    ├── v3_test_results.csv
    ├── v3_scaler.pkl
    ├── v3_label_encoder.pkl
    ├── T10_ablation_study.csv
    ├── T11_statistical_significance.csv
    └── novel_metrics.csv
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

The IoV-SecFusion dataset merges UNSW-NB15, CICIDS2017 and CICIDS2018.

| Dataset      | Source |
|-------------|--------|
| UNSW-NB15   | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CICIDS 2017 | https://www.unb.ca/cic/datasets/ids-2017.html |
| CICIDS 2018 | https://www.unb.ca/cic/datasets/ids-2018.html |

Merge the three sources into a single CSV with a `Label` column containing
the 13 attack class names. The expected shape is approximately
**278,859 rows × 79 features**.

### 3. Train the model

```bash
python train.py \
    --data       IoV_SecFusion_Dataset.csv \
    --output_dir ./outputs
```

This will:
- Split 80/20 (stratified) **before** any preprocessing
- Fit `StandardScaler` on train split only (no feature selection)
- Reshape features to `(N, 1, features)` for LSTM
- Run 5-fold stratified CV
- Evaluate on the held-out 20%
- Save model weights, scaler, label encoder, and result CSVs

**Expected results** (matches paper):

| Metric    | Holdout          | 5-Fold CV          |
|-----------|------------------|--------------------|
| Accuracy  | 91.49%           | 90.67 ± 0.51%      |
| Precision | 94.88%           | —                  |
| F1-Score  | 92.79%           | 91.94 ± 0.51%      |
| AUC-ROC   | 99.61%           | 94.94 ± 0.29%      |

### 4. Run ablation study

```bash
python ablations.py \
    --data       IoV_SecFusion_Dataset.csv \
    --output_dir ./outputs
```

Retrains 5 model variants (w/o Attention, w/o FL, w/o Blockchain Audit,
w/o Class Weighting, LSTM-only) with 5-fold CV and runs Wilcoxon
signed-rank tests on the fold-level F1 arrays.

### 5. Compute novel metrics

```bash
python metrics.py \
    --data       IoV_SecFusion_Dataset.csv \
    --output_dir ./outputs
```

Requires `train.py` to have been run first. Computes ARS, GSPP/PPI, CLR,
DEM from the trained model and saves to `novel_metrics.csv`.

---

## Architecture

```
Input  (batch × 1 × num_features)
     │
     ▼
BiLSTM (hidden=128, num_layers=2, bidirectional) → (batch × 1 × 256)
     │
     ▼
AttentionLayer: Linear(256→1) → softmax over time → weighted sum
     │
     ▼
context (batch × 256)
     │
     ▼
LayerNorm(256)
     │
     ▼
FC(256→128) → ReLU → Dropout(0.3)
     │
     ▼
FC(128→64) → ReLU → Dropout(0.3)
     │
     ▼
FC(64→num_classes) → logits
```

**Loss**: weighted `CrossEntropyLoss` with class weights from
`sklearn.utils.class_weight.compute_class_weight('balanced')`  
**Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)  
**Batch size**: 32  
**Early stopping**: patience=10 epochs  
**Gradient clipping**: max_norm=1.0

---

## Reproducibility notes

- All random seeds fixed to `42`
- **Pipeline order**: split first → scale (fit on train only) → CV
- No feature selection is applied; all numeric features are used
- 5-fold CV uses the same pre-fitted scaler across all folds
- Holdout AUC uses softmax probabilities; CV AUC uses argmax predictions
  (matches the original training code exactly)
- Ablation experiments use the same CV splits as the full model for
  paired Wilcoxon tests

---

## Citation

```bibtex
@article{bcflids2025,
  title   = {BC-FLIDS: A Blockchain-Enhanced Federated Learning Framework
             for Intrusion Detection in Internet of Vehicles},
  author  = {S, Aarthi and others},
  journal = {<journal>},
  year    = {2025}
}
```

---

## License

This code is released for research reproducibility. See `LICENSE` for details.
