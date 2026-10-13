"""FBG tilt regression lab. Run: .venv/bin/python Md_Minha_Uddin_2401CS39_lab9.py"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / "FBG biaxial tilt sensor.xlsx"
SEED = 42
FEATURES = {"F1": ["d1", "d2"], "F2": ["diff"], "F3": ["avg"], "F4": ["diff", "avg"]}


def load_data(sheet):
    df = pd.read_excel(SOURCE, sheet_name=sheet, header=None, usecols="A:G").dropna(how="all")
    df.columns = ["angle", "fbg1", "fbg2", "d1", "d2", "given_diff", "given_avg"]
    df = df.apply(pd.to_numeric, errors="raise")
    assert len(df) == 37 and df.notna().all().all()
    df["diff"] = df.d1 - df.d2
    df["avg"] = (df.d1 + df.d2) / 2
    assert np.allclose(df["diff"], df.given_diff, atol=1e-6)
    assert np.allclose(df["avg"], df.given_avg, atol=1e-6)
    return df.reset_index(drop=True)


def specs():
    # One candidate row is one reported hyperparameter configuration.
    rows = [("SLR", None, LinearRegression(), ["F2", "F3"]),
            ("MLR", None, LinearRegression(), ["F1", "F4"])]
    for degree in (2, 3):
        rows.append(("Polynomial", f"degree={degree}", Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)), ("model", LinearRegression())]), list(FEATURES)))
    for alpha in (0.1, 1, 10):
        for name, cls in (("Ridge", Ridge), ("Lasso", Lasso)):
            rows.append((name, f"alpha={alpha}", cls(alpha=alpha, max_iter=20000), list(FEATURES)))
    for alpha in (0.1, 1):
        rows.append(("Elastic Net", f"alpha={alpha}, l1_ratio=0.5", ElasticNet(alpha=alpha, l1_ratio=.5, max_iter=20000), list(FEATURES)))
    for kernel in ("linear", "rbf"):
        rows.append(("SVR", f"kernel={kernel}, C=10, epsilon=0.1", SVR(kernel=kernel, C=10, epsilon=.1), list(FEATURES)))
    for depth in (2, 3, 5):
        rows.append(("Decision Tree", f"max_depth={depth}", DecisionTreeRegressor(max_depth=depth, random_state=SEED), list(FEATURES)))
    for trees in (50, 100):
        rows.append(("Random Forest", f"n_estimators={trees}", RandomForestRegressor(n_estimators=trees, random_state=SEED, n_jobs=1), list(FEATURES)))
    for k in (3, 5, 7):
        rows.append(("KNN", f"n_neighbors={k}", KNeighborsRegressor(n_neighbors=k), list(FEATURES)))
    for lr in (.01, .1):
        rows.append(("Gradient Boosting", f"learning_rate={lr}, n_estimators=100", GradientBoostingRegressor(learning_rate=lr, n_estimators=100, random_state=SEED), list(FEATURES)))
    for kernel, obj in (("RBF", RBF(1)), ("Matern", Matern(1.5, nu=1.5))):
        rows.append(("GPR", f"kernel={kernel}", GaussianProcessRegressor(kernel=obj, normalize_y=True, optimizer=None, alpha=1e-2), list(FEATURES)))
    return rows


def model_pipe(estimator):
    # Scaler is inside the fold; SLR/MLR and tree models also use the same path.
    return Pipeline([("scale", StandardScaler()), ("model", clone(estimator))])


def evaluate(df, axis):
    groups, y = df.angle.to_numpy(), df.angle.to_numpy()
    outer = GroupKFold(5)
    records, predictions, candidate_folds = [], [], []
    for name, config, estimator, feature_options in specs():
        oof = np.full(len(df), np.nan)
        chosen, fit_seconds = [], []
        for fold, (train, test) in enumerate(outer.split(df, y, groups), 1):
            inner = GroupKFold(4)
            best = None
            # Feature choice happens only on the outer training portion.
            for feature in feature_options:
                x = df[FEATURES[feature]].to_numpy()
                errors = []
                for itrain, ival in inner.split(x[train], y[train], groups[train]):
                    m = model_pipe(estimator)
                    m.fit(x[train][itrain], y[train][itrain])
                    errors.extend(zip(np.repeat(fold, len(ival)), y[train][ival] - m.predict(x[train][ival])))
                rmse = np.sqrt(np.mean([e * e for _, e in errors]))
                if best is None or rmse < best[0]:
                    best = (rmse, feature)
            feature = best[1]
            x = df[FEATURES[feature]].to_numpy()
            model = model_pipe(estimator)
            start = time.perf_counter()
            model.fit(x[train], y[train])
            fit_seconds.append(time.perf_counter() - start)
            oof[test] = model.predict(x[test])
            chosen.append(feature)
            candidate_folds.append({"model": name, "configuration": config or "default", "fold": fold,
                                    "inner_rmse": best[0], "feature": feature, "test": test.copy(),
                                    "prediction": oof[test].copy(), "train_s": fit_seconds[-1]})
            for idx in test:
                predictions.append({"axis": axis, "model": name, "configuration": config or "default", "fold": fold,
                                    "row": int(idx + 1), "angle": float(y[idx]), "prediction": float(oof[idx]),
                                    "residual": float(y[idx] - oof[idx]), "feature": feature})
        assert np.isfinite(oof).all()
        records.append({"axis": axis, "model": name, "configuration": config or "default",
                        "feature_mode": Counter(chosen).most_common(1)[0][0],
                        "feature_folds": ",".join(chosen), "MAE": mean_absolute_error(y, oof),
                        "MSE": mean_squared_error(y, oof), "RMSE": np.sqrt(mean_squared_error(y, oof)),
                        "R2": r2_score(y, oof), "train_s": np.mean(fit_seconds)})
    # Report one fair, nested-tuned result per family. The candidates above also give
    # the requested hyperparameter comparison, but their outer scores are not used to tune.
    for name in dict.fromkeys(row[0] for row in specs()):
        oof = np.full(len(df), np.nan)
        choices, chosen_features, fit_seconds = [], [], []
        for fold in range(1, 6):
            candidates = [r for r in candidate_folds if r["model"] == name and r["fold"] == fold]
            winner = min(candidates, key=lambda r: r["inner_rmse"])
            oof[winner["test"]] = winner["prediction"]
            choices.append(winner["configuration"])
            chosen_features.append(winner["feature"])
            fit_seconds.append(winner["train_s"])
            for idx in winner["test"]:
                predictions.append({"axis": axis, "model": name, "configuration": "nested-tuned", "fold": fold,
                                    "row": int(idx + 1), "angle": float(y[idx]), "prediction": float(oof[idx]),
                                    "residual": float(y[idx] - oof[idx]), "feature": winner["feature"],
                                    "selected_configuration": winner["configuration"]})
        assert np.isfinite(oof).all()
        records.append({"axis": axis, "model": name, "configuration": "nested-tuned",
                        "feature_mode": Counter(chosen_features).most_common(1)[0][0],
                        "feature_folds": ",".join(chosen_features), "configuration_folds": "; ".join(choices),
                        "MAE": mean_absolute_error(y, oof), "MSE": mean_squared_error(y, oof),
                        "RMSE": np.sqrt(mean_squared_error(y, oof)), "R2": r2_score(y, oof),
                        "train_s": np.mean(fit_seconds)})
    return pd.DataFrame(records), pd.DataFrame(predictions)


def exploratory(df, axis):
    desc = df.describe().T.rename(columns={"50%": "median"})
    desc["variance"] = df.var(numeric_only=True)
    desc.to_csv(OUT / f"{axis}_descriptive.csv")
    summary = {"observations": len(df), "input_features": 2, "missing_cells": int(df.isna().sum().sum()),
               "duplicate_rows": int(df.duplicated(subset=["angle", "fbg1", "fbg2", "d1", "d2"]).sum()),
               "repeated_angles": {str(k): int(v) for k, v in df.angle.value_counts().items() if v > 1},
               "angle_min": float(df.angle.min()), "angle_max": float(df.angle.max())}
    correlations = df[["d1", "d2", "diff", "avg", "angle"]].corr()["angle"]
    correlations.to_csv(OUT / f"{axis}_correlations.csv")
    # The feature comparison is descriptive; final model selection stays nested.
    comp = []
    for key, cols in FEATURES.items():
        pred = np.full(len(df), np.nan)
        for tr, te in GroupKFold(5).split(df, df.angle, df.angle):
            m = model_pipe(LinearRegression()).fit(df.iloc[tr][cols], df.angle.iloc[tr])
            pred[te] = m.predict(df.iloc[te][cols])
        comp.append({"axis": axis, "feature": key, "columns": "+".join(cols), "linear_grouped_RMSE": np.sqrt(mean_squared_error(df.angle, pred))})
    return summary, pd.DataFrame(comp)


def plot_exploration(data):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, (axis, df) in enumerate(data.items()):
        ax = axes[i]
        for c in ("fbg1", "fbg2"):
            ax[0].scatter(df.angle, df[c], s=15, label=c)
        ax[0].set(title=f"{axis}: raw FBG responses", xlabel="Angle (deg)", ylabel="Sensor reading")
        ax[0].legend(fontsize=7)
        sns.heatmap(df[["angle", "d1", "d2", "diff", "avg"]].corr(), vmin=-1, vmax=1, cmap="coolwarm", ax=ax[1], cbar=False, annot=True, fmt=".1f", annot_kws={"size":6})
        ax[1].set_title(f"{axis}: Pearson r")
        sns.boxplot(data=df[["d1", "d2", "diff", "avg"]], ax=ax[2], color="#9ac7c9")
        ax[2].set_title(f"{axis}: outlier check")
        z = StandardScaler().fit_transform(df[["d1", "d2"]])
        mm = MinMaxScaler().fit_transform(df[["d1", "d2"]])
        ax[3].hist(df.d1, bins=9, alpha=.5, label="raw d1")
        ax[3].hist(z[:, 0], bins=9, alpha=.5, label="standard d1")
        ax[3].hist(mm[:, 0], bins=9, alpha=.5, label="min-max d1")
        ax[3].set_title(f"{axis}: scaling demo")
        ax[3].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "exploration.png", dpi=160); plt.close(fig)


def plot_results(results, preds):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, axis in enumerate(("axial", "transverse")):
        best = results[(results.axis == axis) & (results.configuration == "nested-tuned")].sort_values("RMSE")
        for j, metric in enumerate(("MAE", "RMSE", "R2", "train_s")):
            ax = axes[i, j]
            ax.barh(best.model, best[metric], color="#457b9d")
            ax.invert_yaxis(); ax.set_title(f"{axis}: {metric}")
            ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    stat = {"SLR", "MLR", "Polynomial", "Ridge", "Lasso", "Elastic Net"}
    for i, axis in enumerate(("axial", "transverse")):
        r = results[(results.axis == axis) & (results.configuration == "nested-tuned")]
        for k, sub in enumerate((r[r.model.isin(stat)], r[~r.model.isin(stat)])):
            winner = sub.loc[sub.RMSE.idxmin()]
            p = preds[(preds.axis == axis) & (preds.model == winner.model) & (preds.configuration == winner.configuration)]
            a, b = axes[i, k*2:k*2+2]
            a.scatter(p.angle, p.prediction, s=20); a.plot([-90, 90], [-90, 90], "k--", lw=1)
            a.set(title=f"{axis}: {winner.model} {winner.configuration}", xlabel="Actual (deg)", ylabel="Predicted (deg)")
            b.scatter(p.angle, p.residual, s=20); b.axhline(0, color="k", ls="--", lw=1)
            b.set(title=f"RMSE {winner.RMSE:.2f} deg", xlabel="Actual (deg)", ylabel="Residual (deg)")
    fig.tight_layout(); fig.savefig(OUT / "predictions_residuals.png", dpi=160); plt.close(fig)


def advanced(data):
    df = data["transverse"]
    rows = []
    for target in ("signed", "absolute"):
        y = df.angle.to_numpy() if target == "signed" else np.abs(df.angle.to_numpy())
        pred = np.full(len(df), np.nan)
        for tr, te in GroupKFold(5).split(df, y, df.angle):
            m = model_pipe(LinearRegression()).fit(df.iloc[tr][["avg"]], y[tr])
            pred[te] = m.predict(df.iloc[te][["avg"]])
        rows.append({"target": target, "model": "linear avg", "RMSE": np.sqrt(mean_squared_error(y, pred)), "MAE": mean_absolute_error(y, pred), "R2": r2_score(y, pred)})
    return pd.DataFrame(rows)


def main():
    data = {"axial": load_data("Experiment axial "), "transverse": load_data("Experiment transverse ")}
    summaries, feature_tables, result_tables, pred_tables = {}, [], [], []
    for axis, df in data.items():
        summaries[axis], ft = exploratory(df, axis)
        feature_tables.append(ft)
        r, p = evaluate(df, axis)
        result_tables.append(r); pred_tables.append(p)
    results, preds = pd.concat(result_tables, ignore_index=True), pd.concat(pred_tables, ignore_index=True)
    results.to_csv(OUT / "all_model_results.csv", index=False)
    preds.to_csv(OUT / "all_observation_predictions.csv", index=False)
    tuned = preds[preds.configuration == "nested-tuned"].copy()
    tuned["absolute_error"] = tuned.residual.abs()
    tuned.sort_values("absolute_error", ascending=False).groupby(["axis", "model"]).head(5).to_csv(OUT / "worst_angle_errors.csv", index=False)
    pd.concat(feature_tables).to_csv(OUT / "feature_comparison.csv", index=False)
    with open(OUT / "dataset_summary.json", "w") as f: json.dump(summaries, f, indent=2)
    advanced(data).to_csv(OUT / "advanced_signed_absolute.csv", index=False)
    plot_exploration(data); plot_results(results, preds)
    print(results.sort_values(["axis", "RMSE"]).groupby("axis").head(5).to_string(index=False))
    print("Wrote results/ files")


if __name__ == "__main__":
    main()
