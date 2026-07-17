# Dataset Download Instructions

This study uses three publicly available network intrusion datasets to build
IoV-SecFusion — a new fused IoV security dataset covering 13 real-world
attack types. The raw datasets are not included in this repository due to
their large file sizes. Please follow the instructions below to download
each dataset separately.

---

## Dataset 1 — UNSW-NB15

**Source:** Australian Centre for Cyber Security, UNSW Canberra
**Year:** 2015
**Size:** ~100 MB (CSV files only)
**Attack Types:** 9 (Fuzzers, DoS, Backdoor, Exploits, Generic, Recon, Shellcode, Worms, Normal)

**Download Link:**
https://research.unsw.edu.au/projects/unsw-nb15-dataset

**Files to Download (CSV folder only):**
- UNSW-NB15_1.csv
- UNSW-NB15_2.csv
- UNSW-NB15_3.csv
- UNSW-NB15_4.csv
- NUSW-NB15_GT.csv
- NUSW-NB15_features.csv

**How to cite:**
Moustafa N, Slay J. UNSW-NB15: a comprehensive data set for network intrusion
detection systems. MilCIS, IEEE, 2015.
DOI: 10.1109/MilCIS.2015.7348942

---

## Dataset 2 — CICIDS2017

**Source:** Canadian Institute for Cybersecurity, University of New Brunswick
**Year:** 2017
**Size:** ~50-100 MB (CSV files)
**Attack Types:** 7 (DoS, DDoS, Brute Force, XSS, SQL Injection, Infiltration, Botnet)

**Download Link (Kaggle — easiest):**
https://www.kaggle.com/datasets/cicdataset/cicids2017

**Files to Download:**
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv
- Monday-WorkingHours.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-WorkingHours.pcap_ISCX.csv

**How to cite:**
Sharafaldin I, Lashkari AH, Ghorbani AA. Toward generating a new intrusion
detection dataset and intrusion traffic characterization. ICISSP, 2018.
DOI: 10.5220/0006639801080116

---

## Dataset 3 — CICIDS2018

**Source:** Canadian Institute for Cybersecurity, University of New Brunswick
**Year:** 2018
**Size:** ~600 MB - 1 GB (selected CSV files only)
**Attack Types:** 7 (Brute Force, DoS, Infiltration, Botnet, DDoS, Web Attacks, SSH)

**Download Link (Kaggle — easiest):**
https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv

**Recommended Files to Download:**
- 02-14-2018.csv (FTP and SSH BruteForce)
- 02-22-2018.csv (Botnet ARES)
- 02-28-2018.csv (Infiltration)
- 03-02-2018.csv (DDoS and DoS attacks)

**How to cite:**
Sharafaldin I, Lashkari AH, Ghorbani AA. Toward generating a new intrusion
detection dataset and intrusion traffic characterization. ICISSP, 2018.
DOI: 10.5220/0006639801080116

---

## After Downloading

Once all three datasets are downloaded, place them in the following structure:

```
datasets/
├── UNSW-NB15/
│   ├── UNSW-NB15_1.csv
│   ├── UNSW-NB15_2.csv
│   ├── UNSW-NB15_3.csv
│   └── UNSW-NB15_4.csv
│
├── CICIDS2017/
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   └── Wednesday-WorkingHours.pcap_ISCX.csv
│
└── CICIDS2018/
    ├── 02-14-2018.csv
    ├── 02-22-2018.csv
    └── 03-02-2018.csv
```

Then update the file paths in `preprocessing/merge_iov_dataset.py`
to match your local folder structure before running.

---

## IoV-SecFusion — Proposed Fused Dataset

After running the preprocessing scripts, a new fused dataset called
IoV-SecFusion will be created automatically. It covers 13 attack types:

| Class | Source Dataset |
|-------|---------------|
| Benign | All three datasets |
| Bot | CICIDS2017 / CICIDS2018 |
| DDoS | CICIDS2017 / CICIDS2018 |
| DoS-GoldenEye | CICIDS2017 |
| DoS-Hulk | CICIDS2017 |
| DoS-Slowhttptest | CICIDS2017 |
| DoS-Slowloris | CICIDS2017 |
| FTP-BruteForce | CICIDS2018 |
| Heartbleed | CICIDS2017 |
| SQL Injection | CICIDS2017 |
| SSH-BruteForce | CICIDS2018 |
| WebAttack-BruteForce | CICIDS2017 |
| WebAttack-XSS | CICIDS2017 |

---

## Notes

- Do not upload the raw dataset CSV files to GitHub due to their large sizes
- The preprocessing scripts handle all cleaning, merging, and balancing
- All preprocessing steps follow Algorithm 1 described in the paper
- Random seed 42 is used throughout for full reproducibility
