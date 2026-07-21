# Reproducibility guide

## Recommended environment

- Python 3.10+
- GPU-enabled XGBoost
- Colab GPU or a CUDA workstation
- Sufficient RAM for the full 4.5-million-row training split

## Run order

1. `notebooks/01_mandatory_start.ipynb`
2. `notebooks/02_zero_start.ipynb`
3. `notebooks/03_paper_comparison.ipynb`

The first two notebooks download the public dataset. The third reads committed result
files and does not require model training.

## Verification

```bash
python -m pip install -r requirements-dev.txt
pytest
python scripts/build_figures.py
```

## Expected headline values

```text
Mandatory start overall WAPE: 33.030551%
Zero start overall WAPE:      29.957163%
```

## Confirmatory feature-selection run

Use `configs/confirmatory_same_reset.json` to rerun selection with a same-reset
selected-set baseline and a small minimum improvement threshold.
