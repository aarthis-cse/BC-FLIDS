# BC-FLIDS: Securing Internet of Vehicles Against Multi-Class Cyber Attacks

## Overview

BC-FLIDS stands for Blockchain-Federated Learning Intrusion Detection System. It was built to solve three core problems in Internet of Vehicles (IoV) security at the same time — privacy, centralization, and multi-class attack detection. The model combines a Federated Learning privacy engine, a Blockchain audit layer, and a BiLSTM with Multi-Head Attention detection model into one complete system, which means raw vehicle data never leaves the vehicle during training and model updates cannot be changed without being detected.

---

## Paper

**Title:** BC-FLIDS: Securing Internet of Vehicles Against Multi-Class Cyber Attacks Using Blockchain-Integrated Federated BiLSTM with Multi-Head Attention

**Journal:** Concurrency and Computation: Practice and Experience (Wiley)

**Status:** Under Review

---

## Key Results

| Metric | CV Mean (5-Fold) | Std Dev | Holdout Test (20%) |
|--------|-----------------|---------|-------------------|
| Accuracy | 90.67% | ±0.51% | 91.49% |
| Precision | 94.11% | ±0.46% | 94.88% |
| Recall | 90.67% | ±0.51% | 91.49% |
| F1-Score | 91.94% | ±0.51% | 92.79% |
| AUC-ROC | 94.94% | ±0.29% | 99.61% |

All improvements over baseline models are statistically significant at p < 0.05 (Wilcoxon signed-rank test).

---

## Model Architecture

BC-FLIDS has three working layers:

**Layer 1 — Federated Learning Engine**
Each vehicle trains a local BiLSTM-Attention model on its own traffic data and only shares model gradients, not raw data, which means vehicle privacy is preserved by design.

**Layer 2 — Blockchain Audit Layer**
Every gradient update is recorded on a lightweight blockchain ledger using Proof of Authority consensus before being combined into the shared model, which means no single node can tamper with the learning process without being detected. Smart contracts automatically isolate compromised vehicles when an attack is detected.

**Layer 3 — BiLSTM with Multi-Head Attention**
A two-layer Bidirectional LSTM with hidden size 128, combined with a four-head attention mechanism, that learns temporal attack patterns from IoV network traffic.

---

## Dataset — IoV-SecFusion

A new fused IoV security dataset was built by combining three publicly available datasets:

| Dataset | Source | Year | Attack Types |
|---------|--------|------|-------------|
| UNSW-NB15 | UNSW Canberra | 2015 | 9 |
| CICIDS2017 | Canadian Institute for Cybersecurity | 2017 | 7 |
| CICIDS2018 | Canadian Institute for Cybersecurity | 2018 | 7 |
| **IoV-SecFusion (Proposed)** | **This Study** | **2025** | **13** |

Please see `datasets/README_datasets.md` for download instructions.

---

## Novel Evaluation Metrics

Four new metrics were introduced in this study:

| Metric | Abbreviation | What It Measures |
|--------|-------------|-----------------|
| Attack Resilience Score | ARS | How fast the system recovers after a successful attack |
| Privacy Preservation Index | PPI | How much sensitive vehicle data is exposed during FL training |
| Consensus Latency Ratio | CLR | Blockchain overhead relative to total detection response time |
| Decentralization Efficiency Metric | DEM | How evenly trust and computation are shared across IoV nodes |

---

## Repository Structure

```
BC-FLIDS-IoV-Security/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── datasets/
│   └── README_datasets.md
│
├── preprocessing/
│   ├── merge_iov_dataset.py
│   ├── clean_labels.py
│   └── verify_dataset.py
│
├── model/
│   └── bc_flids_v3.py
│
├── evaluation/
│   ├── generate_all_graphs.py
│   ├── generate_all_tables.py
│   ├── generate_generalization.py
│   └── generate_ablation_significance.py
│
├── results/
│   ├── v3_cv_results.csv
│   └── v3_test_results.csv
│
└── graphs/
    ├── G1_train_val_curves.png
    ├── G2_boxplot_cv.png
    ├── G3_confusion_matrix.png
    ├── G4_roc_auc_curves.png
    ├── G5_tsne_clusters.png
    ├── G6_baseline_comparison.png
    ├── G7_precision_recall.png
    ├── G8_radar_chart.png
    ├── G9_attack_resilience_score.png
    ├── G10_privacy_preservation_index.png
    ├── G11_consensus_latency_ratio.png
    ├── G12_class_distribution.png
    ├── G13_perclass_f1.png
    ├── G14_shap_importance.png
    ├── G15_time_vs_accuracy.png
    ├── G16_generalization_analysis.png
    ├── G17_ablation_study.png
    └── G18_training_time.png
```

---

## How to Run

### Step 1 — Install requirements
```bash
pip install -r requirements.txt
```

### Step 2 — Download datasets
Follow the instructions in `datasets/README_datasets.md`

### Step 3 — Merge and clean datasets
```bash
python preprocessing/merge_iov_dataset.py
python preprocessing/clean_labels.py
```

### Step 4 — Train BC-FLIDS model
```bash
python model/bc_flids_v3.py
```

### Step 5 — Generate all graphs and tables
```bash
python evaluation/generate_all_graphs.py
python evaluation/generate_all_tables.py
python evaluation/generate_generalization.py
python evaluation/generate_ablation_significance.py
```

---

## Requirements

- Python 3.10 or above
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SHAP
- SciPy
- imbalanced-learn

See `requirements.txt` for exact versions.

---

## Hyperparameter Settings

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Weight Decay | 1e-4 |
| Batch Size | 128 |
| Max Epochs | 100 |
| Early Stopping Patience | 15 |
| BiLSTM Layers | 2 |
| Hidden Size | 128 |
| Attention Heads | 4 |
| Dropout | 0.3 |
| Loss Function | Focal Loss (γ=2.0, α=1.0) |
| Cross Validation | Stratified 5-Fold |
| Holdout Test | 20% |

---

## Baseline Models

BC-FLIDS was compared against four baseline models:

| Model | Type |
|-------|------|
| Isolation Forest | Traditional Machine Learning |
| CNN-LSTM | Deep Learning |
| BC-Only IDS | Blockchain without Federated Learning |
| FL-Only IDS | Federated Learning without Blockchain |

---

## Citation

If you use this code or dataset in your research, please cite our paper:

```
Citation will be added after publication.
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions about this repository or the paper, please open an issue on GitHub.
