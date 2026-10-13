# FBG tilt regression lab

This is Md Minha Uddin's (2401CS39) work for Experiment 06 (13 October 2026).
Public code: https://github.com/mdminhaj-2106/ML_Lab9
The submission file is `Md_Minha_Uddin_2401CS39_lab9.pdf` (six A4 pages).
`Md_Minha_Uddin_2401CS39_lab9.ipynb` is a short walkthrough;
`Md_Minha_Uddin_2401CS39_lab9.py` runs the full experiment; `make_report.py`
rebuilds the PDF from the saved results.

## Data

Download **FBG biaxial tilt sensor.xlsx** from Kuo Li's [Mendeley Data record](https://data.mendeley.com/datasets/f4csv2xjk6/1)
(DOI 10.17632/f4csv2xjk6.1, CC BY 4.0) and put it next to the Python script.
The workbook supplied for this lab is already in this folder. Only the two experimental
sheets are used; the first row is data, and columns A:G contain the 37 observations.

## Reproduce

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python Md_Minha_Uddin_2401CS39_lab9.py
.venv/bin/python make_report.py
```

The run saves descriptive statistics, feature comparisons, metrics for every model
configuration, all outer-fold predictions, the largest per-angle errors, plots, and
the signed-versus-absolute transverse result in `results/`.

Validation uses five grouped outer folds and four grouped inner folds, grouped by
angle. Scalers and feature selection are fitted inside training folds. The `nested-tuned`
rows are the fair final family comparison. The other rows show the requested fixed
hyperparameter experiments.
