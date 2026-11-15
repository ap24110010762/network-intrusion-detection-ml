# Explainable ML for Network Intrusion Detection and Attack Classification

Course project (SRM University-AP). Detects malicious network flows in the CIC-IDS2017 dataset
(binary: BENIGN vs ATTACK) and names the attack type (10 classes), with SHAP explanations
and a Streamlit demo.

## Results (locked test set, 499,596 flows, scored once)

| Task | Result |
|---|---|
| Binary (XGBoost) | recall 0.9986, precision 0.9968, F1 0.9977, ROC-AUC 0.99997; 116 missed attacks, 275 false alarms (0.066% of normal flows) |
| 10-class (XGBoost + SMOTE) | accuracy 0.9992, macro-F1 0.9566; weakest class: Bot (F1 0.84) |

Heartbleed (2 test flows) and Infiltration (7 test flows) are too small for reliable estimates.

## Method in short

1. Cleaning: removed NaN/inf rows, 329,896 duplicate rows, 1 duplicate and 8 constant columns; 15 raw labels grouped into 10 classes.
2. Stratified 80/20 split; the test set stayed locked until the final evaluation.
3. Six models compared with stratified 5-fold cross-validation (mean +/- std); signed-log transform + scaling for linear/KNN/SVM models.
4. Class imbalance: class weights, undersampling and SMOTE compared; SMOTE used for the 10-class model (training folds only).
5. Randomized hyper-parameter search (little gain, models near their ceiling).
6. Feature sets compared: 48 correlation-filtered features kept (the 25-feature set failed on rare classes).
7. Explainability: XGBoost gain importance and SHAP (global, per-class, single-flow).

## Limitations (please read)

- One lab testbed (CIC-IDS2017); results may not transfer to other networks.
- SHAP shows reliance on destination port and TCP window size (Init_Win_bytes_*). Removing both lowers binary F1 to about 0.990 and Bot F1 to about 0.71 in cross-validation.
- Most Web Attack flows have zero payload bytes, so that class is probably recognised by flow/lab characteristics, not by attack content.
- Random split: flows from the same attack sessions appear in train and test.

## Run the demo (Windows)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app/app.py

## Folders

- `notebooks/` - all experiments in order (01 to 09)
- `src/` - helper code (`preprocessing.py` is needed to load the saved models)
- `models/` - final binary and 10-class XGBoost models (portable JSON files used by the app; joblib pipelines kept for reference)
- `results/` - tables, figures, experiment log
- `app/` - Streamlit app and a small demo sample

The dataset is not included. Download CIC-IDS2017 from the Canadian Institute for Cybersecurity website.
