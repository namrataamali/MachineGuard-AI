# MachineGuard AI — Predictive Maintenance & Smart Recommendations

A dark, cinematic Streamlit dashboard for predicting industrial machine failures, reviewing risk levels, generating illustrative maintenance recommendations, and analyzing CSV batches.

## Quick start (Windows)

1. Install Python 3.10 or later and ensure `py` or `python` works in your terminal.
2. Extract the ZIP.
3. Double-click `START_WINDOWS.bat` **or** open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On first launch, if no saved model exists, the app downloads the **UCI AI4I 2020 synthetic dataset**, trains the Random Forest model, and saves the model files locally. Internet access is needed for this first launch. Later launches reuse the saved model.

If you already trained the model in Colab, put `machineguard_model.pkl` and `feature_names.pkl` next to `app.py` before launching. Both files must come from the same training run.

## Features
- Cinematic dashboard and machine health assessment
- Model-estimated failure score and low/medium/high risk labels
- Rule-based maintenance suggestions
- Batch CSV upload, risk visualization, and downloadable report
- Downloadable sample CSV

## Input format
Use `Type` (L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`.

## Important limitations
The dataset is synthetic. Risk bands and inspection recommendations are illustrative, not validated maintenance rules. The 25% classification threshold was explored on a test set, so a proper validation/test separation is required before claiming unbiased performance. The model's output scores are not calibrated real-world failure probabilities. Do not use this demonstration to make safety-critical maintenance decisions.

Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
