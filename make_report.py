"""Build the submission PDF from the executed lab results."""
from pathlib import Path
import json
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).parent
OUT = ROOT / "results"
PDF = ROOT / "Md_Minha_Uddin_2401CS39_lab9.pdf"
R = pd.read_csv(OUT / "all_model_results.csv")
F = pd.read_csv(OUT / "feature_comparison.csv")
A = pd.read_csv(OUT / "advanced_signed_absolute.csv")
S = json.loads((OUT / "dataset_summary.json").read_text())
P = pd.read_csv(OUT / "all_observation_predictions.csv")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleLab", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=19, textColor=colors.HexColor("#173b52"), spaceAfter=5))
styles.add(ParagraphStyle(name="HeadingLab", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=colors.HexColor("#173b52"), spaceBefore=7, spaceAfter=3))
styles.add(ParagraphStyle(name="BodyLab", parent=styles["Normal"], fontName="Helvetica", fontSize=8.4, leading=11.2, spaceAfter=4))
styles.add(ParagraphStyle(name="SmallLab", parent=styles["Normal"], fontName="Helvetica", fontSize=7.2, leading=9.3, spaceAfter=3))

def h(s): return Paragraph(s, styles["HeadingLab"])
def p(s, small=False): return Paragraph(s, styles["SmallLab" if small else "BodyLab"])

def t(rows, widths, size=7, pad=2):
    x = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    x.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("LEADING", (0, 0), (-1, -1), size + 1.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceaf0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#aebfc8")),
    ]))
    return x

def fig(name, width=515):
    im = PILImage.open(OUT / name)
    return Image(str(OUT / name), width=width, height=width * im.height / im.width)

def family_table(axis):
    sub = R[(R.axis == axis) & (R.configuration == "nested-tuned")].sort_values("RMSE")
    rows = [["Model", "MAE", "MSE", "RMSE", "R2", "Fit ms", "Feat."]]
    rows += [[x.model, f"{x.MAE:.2f}", f"{x.MSE:.1f}", f"{x.RMSE:.2f}", f"{x.R2:.3f}", f"{x.train_s*1000:.2f}", x.feature_mode] for x in sub.itertuples()]
    return t(rows, [127, 58, 63, 58, 55, 68, 87], size=7, pad=2)

def config_table(axis):
    sub = R[(R.axis == axis) & (R.configuration != "nested-tuned")]
    rows = [["Model", "Configuration", "RMSE", "R2"]]
    for x in sub.itertuples():
        config = (x.configuration.replace("learning_rate", "lr").replace("n_estimators", "trees")
                  .replace("n_neighbors", "k").replace("max_depth", "depth")
                  .replace("l1_ratio", "l1").replace("kernel", "kernel")
                  .replace("epsilon", "eps"))
        rows.append([x.model, config, f"{x.RMSE:.2f}", f"{x.R2:.3f}"])
    return t(rows, [75, 112, 34, 25], size=7.0, pad=5.2)

best = R[R.configuration == "nested-tuned"]
stat = {"SLR", "MLR", "Polynomial", "Ridge", "Lasso", "Elastic Net"}
winner = {}
for axis in ("axial", "transverse"):
    sub = best[best.axis == axis]
    winner[axis] = {"stat": sub[sub.model.isin(stat)].sort_values("RMSE").iloc[0],
                    "ml": sub[~sub.model.isin(stat)].sort_values("RMSE").iloc[0]}

story = [
    Paragraph("FBG tilt angle prediction", styles["TitleLab"]),
    p("<b>Md Minha Uddin</b> | 2401CS39 | <b>ML Lab 9</b> | 13 October 2026"),
    h("Aim and data"),
    p("I predicted axial and transverse tilt separately from the two FBG wavelength changes. I used all 37 readings in each experimental sheet, including row 1. The theoretical sheet and extra chart cells were not used. No synthetic rows or imputations."),
    t([["Axis", "Rows", "Angle range", "Missing", "Exact repeats", "Repeated angles"],
       ["Axial", "37", "-90 to +90 deg", "0", str(S["axial"]["duplicate_rows"]), "17 values"],
       ["Transverse", "37", "-90 to +90 deg", "0", str(S["transverse"]["duplicate_rows"]), "17 values"]],
      [66, 45, 105, 60, 110, 131]),
    h("Method and one end-to-end model"),
    p("F1 uses both wavelength changes, F2 their difference, F3 their average, and F4 difference plus average. I checked F2/F3 against the workbook columns. I used five outer GroupKFold splits by angle. Four grouped inner folds selected features and hyperparameters from training rows only. StandardScaler is inside each fitted pipeline. Metrics come from the joined outer-fold predictions."),
    p("<b>Gaussian process example:</b> D/E changes &rarr; F1/F2/F3/F4 &rarr; grouped split &rarr; inner RMSE chooses feature and RBF/Matérn kernel &rarr; fit scaler and GPR on outer training rows &rarr; predict held-out angles &rarr; repeat five times and calculate MAE, MSE, RMSE, R2. Seed 42 is used for stochastic trees."),
    h("Sensor responses, correlations, outliers and scaling"),
    fig("exploration.png"),
    p("Axial difference tracks signed angle. In transverse tilt, both FBGs mostly rise with magnitude, so the sign is harder. Boxplots flag extremes but I kept all recorded measurements. Scaling shown here is only a demo; evaluation scalers fit training folds only.", True),
    p("Source: Kuo Li, <i>FBG biaxial tilt sensor using one optical fiber</i>, Mendeley Data, doi:10.17632/f4csv2xjk6.1, CC BY 4.0.", True),
    PageBreak(),
]

feature_rows = [["Axis", "F1 pair", "F2 diff", "F3 average", "F4 diff+avg"]]
feature_rows += [[axis.capitalize()] + [f"{F[(F.axis == axis) & (F.feature == key)].iloc[0].linear_grouped_RMSE:.2f}" for key in ("F1", "F2", "F3", "F4")] for axis in ("axial", "transverse")]
story += [
    h("Feature check: grouped linear RMSE (degrees)"),
    t(feature_rows, [87, 107, 107, 107, 107]),
    p("F2 is strongest for axial linear prediction. Signed transverse prediction needs information from both FBGs; F1 and F4 contain the same two underlying measurements.", True),
    h("Nested-tuned performance: axial"), family_table("axial"),
    h("Nested-tuned performance: signed transverse"), family_table("transverse"),
    h("Accuracy and fit-time comparison"), fig("model_comparison.png"),
    p("All bars use outer-fold predictions after inner-fold selection. Fit time is the mean final training fit per fold, excluding search. On 37 rows it is only a rough speed indicator.", True),
    PageBreak(),
]

configs = Table([[config_table("axial"), "", config_table("transverse")]], colWidths=[246, 24, 246], hAlign="LEFT")
configs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
top = P[(P.configuration == "nested-tuned") & (P.model == "GPR")].copy()
top["absolute_error"] = top.residual.abs()
worst = top.sort_values("absolute_error", ascending=False).groupby("axis").head(4)
worst_rows = [["Axis", "Actual", "Predicted", "Abs. error", "Fold"]]
worst_rows += [[x.axis.capitalize(), f"{x.angle:.0f}", f"{x.prediction:.1f}", f"{x.absolute_error:.1f}", str(x.fold)] for x in worst.itertuples()]
story += [
    h("Hyperparameter checks: every requested setting"),
    p("Fixed-setting scores below also choose features in the inner training folds. The final family scores on page 2 use inner-fold hyperparameter selection too; I did not choose a winner by looking at outer test scores.", True),
    t([["Axial: RMSE and R2", "Transverse: RMSE and R2"]], [258, 258], size=8),
    configs,
    h("Where the best ML model missed"),
    t(worst_rows, [102, 96, 106, 112, 100], size=7),
    p("These are the four largest held-out absolute errors for GPR on each axis. They show which individual angles are hard rather than hiding them in one average.", True),
    PageBreak(),
]

signed, absolute = A.iloc[0], A.iloc[1]
story += [
    h("Actual versus predicted angles and residuals"),
    fig("predictions_residuals.png"),
    p(f"Best statistical model: {winner['axial']['stat'].model} on axial ({winner['axial']['stat'].RMSE:.2f} deg RMSE), {winner['transverse']['stat'].model} on transverse ({winner['transverse']['stat'].RMSE:.2f} deg). Best conventional ML model: {winner['axial']['ml'].model} on axial ({winner['axial']['ml'].RMSE:.2f} deg), {winner['transverse']['ml'].model} on transverse ({winner['transverse']['ml'].RMSE:.2f} deg). These are held-out predictions."),
    h("Advanced task: signed versus absolute transverse tilt"),
    t([["Target", "Input/model", "MAE", "RMSE", "R2"],
       ["Signed angle", "Average change / linear", f"{signed.MAE:.2f}", f"{signed.RMSE:.2f}", f"{signed.R2:.3f}"],
       ["Absolute angle", "Average change / linear", f"{absolute.MAE:.2f}", f"{absolute.RMSE:.2f}", f"{absolute.R2:.3f}"]],
      [110, 178, 76, 76, 76]),
    p("The average change tells me the amount of transverse tilt much better than its direction. Positive and negative angles often produce similar readings. GPR finds some direction clues from both channels, but I would check those on more sensor runs before trusting the sign."),
    h("What I take from it"),
    p("Degree 3 polynomial improves axial accuracy over a straight line, but added complexity is not always useful: deeper trees and more forest trees do not consistently win. Feature choice matters: axial F2 is useful alone, while signed transverse F3 is weak alone. Axial errors are smaller because its differential response tracks sign more directly."),
    p("For best accuracy on these measurements I would use Matérn GPR. For a simple real-time axial monitor, I would also consider linear regression on the differential signal because it is easy to calibrate and fast. For transverse monitoring I would report magnitude unless direction is separately validated. This is still only 37 readings per axis."),
    p('<link href="https://github.com/mdminhaj-2106/ML_Lab9" color="blue">github.com/mdminhaj-2106/ML_Lab9</link> has the code, notebook and row-wise results.', True),
]

def footer(canvas, doc):
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#657783"))
    canvas.drawString(38, 19, "Md Minha Uddin | 2401CS39 | ML Lab 9")
    canvas.drawRightString(A4[0] - 38, 19, str(doc.page))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=38, rightMargin=38, topMargin=33, bottomMargin=32)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(PDF)
