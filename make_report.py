"""Build the six-page submission PDF from the executed lab results."""
from pathlib import Path
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).parent
OUT = ROOT / "results"
R = pd.read_csv(OUT / "all_model_results.csv")
F = pd.read_csv(OUT / "feature_comparison.csv")
A = pd.read_csv(OUT / "advanced_signed_absolute.csv")
S = json.loads((OUT / "dataset_summary.json").read_text())
PDF = ROOT / "Md_Minha_Uddin_2401CS39_lab9.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="LabTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=colors.HexColor("#12334b"), spaceAfter=9))
styles.add(ParagraphStyle(name="LabH", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=colors.HexColor("#12334b"), spaceBefore=7, spaceAfter=4))
styles.add(ParagraphStyle(name="LabBody", parent=styles["Normal"], fontName="Helvetica", fontSize=8.4, leading=11.3, spaceAfter=5))
styles.add(ParagraphStyle(name="LabSmall", parent=styles["Normal"], fontName="Helvetica", fontSize=7.1, leading=9.3, spaceAfter=3))


def p(text, small=False): return Paragraph(text, styles["LabSmall" if small else "LabBody"])
def h(text): return Paragraph(text, styles["LabH"])


def table(rows, widths=None, size=7.0, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), size),
                ("LEADING", (0, 0), (-1, -1), size+2), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#b5c4cc"))]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9ef")),
                     ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0: commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f8fa")))
    t.setStyle(TableStyle(commands))
    return t


def figure(name, width=516):
    from PIL import Image as PILImage
    im = PILImage.open(OUT / name)
    return Image(str(OUT / name), width=width, height=width * im.height / im.width)


story = []
story += [Paragraph("FBG tilt angle prediction", styles["LabTitle"]),
          p("Md Minha Uddin | Roll no. 2401CS39 | Machine Learning Laboratory, Experiment 06 | 13 October 2026"),
          h("What I did"),
          p("I used the two experimental sheets from the given workbook. Each has 37 actual readings, including row 1. The theoretical sheet and the extra chart data on the right of the workbook were left out. The target is angle in degrees; the starting inputs are the two wavelength changes (columns D and E)."),
          p("I tried the original pair (F1), their difference (F2), their average (F3), and difference plus average (F4). I checked the supplied F and G columns against my calculations; they match. The four choices are not four independent sources of information: F1 and F4 are just invertible ways of writing the same two readings."),
          h("Validation and leakage control"),
          p("For each axis I used five GroupKFold outer folds, grouped by angle so repeated measurements at one angle stay together. Within each outer training set, four grouped inner folds selected the feature configuration and model hyperparameter. StandardScaler sits inside each fitted pipeline. I joined the 37 outer-fold predictions before calculating MAE, MSE, RMSE and R-squared. Training time is the mean final fit time per outer fold, excluding the tuning search; it is only a rough comparison on this tiny dataset."),
          h("One model from start to finish: GPR"),
          p("<b>Flow:</b> experimental D/E changes &rarr; compute F1/F2/F3/F4 &rarr; grouped outer training/test split &rarr; grouped inner selection of features and RBF/Matérn kernel &rarr; fit scaler on training rows &rarr; fit Gaussian process &rarr; predict held-out angles &rarr; collect errors. Repeat for all five outer folds."),
          p("<b>Algorithm:</b> 1) Make five angle groups. 2) For each outer fold, score every allowed feature/kernel pair in four inner grouped folds using RMSE. 3) Refit the best pair on the whole outer training split. 4) Predict the untouched outer test split. 5) Aggregate all 37 predictions. Seed 42 is used for the stochastic tree models."),
          h("Dataset checks"),
          table([["Axis", "Rows", "Angle range", "Missing", "Exact duplicate rows", "Repeated angles"],
                 ["Axial", "37", "-90 to +90 deg", "0", str(S["axial"]["duplicate_rows"]), "17 angle values"],
                 ["Transverse", "37", "-90 to +90 deg", "0", str(S["transverse"]["duplicate_rows"]), "17 angle values"]],
                [68, 42, 105, 55, 106, 116]),
          Spacer(1, 5),
          p("The axial sheet has two exact repeats; I kept them because they are recorded observations. Most non-endpoint angles have two readings and zero has three. No rows were added or imputed. The full descriptive statistics and every row-wise prediction are saved with the code.", True),
          p("Data: Kuo Li, <i>FBG biaxial tilt sensor using one optical fiber</i>, Mendeley Data, doi:10.17632/f4csv2xjk6.1 (CC BY 4.0).", True),
          PageBreak()]

story += [h("Feature check (five grouped folds; linear model only)"),
          table([["Axis", "F1 pair", "F2 diff", "F3 average", "F4 diff+avg"]] +
                [[axis.capitalize()] + [f"{F[(F.axis==axis)&(F.feature==key)].iloc[0].linear_grouped_RMSE:.2f}" for key in ("F1","F2","F3","F4")] for axis in ("axial","transverse")],
                [85, 105, 105, 105, 105]),
          p("Values are RMSE in degrees. Axial F2 is already strong by itself. For signed transverse angle, even the best linear pair remains weak; average alone mostly describes tilt magnitude.", True),
          h("Nested-tuned model family comparison: axial"),
          table([["Model", "MAE", "MSE", "RMSE", "R2", "Fit ms", "Feature"]] +
                [[x.model, f"{x.MAE:.2f}", f"{x.MSE:.1f}", f"{x.RMSE:.2f}", f"{x.R2:.3f}", f"{x.train_s*1000:.2f}", x.feature_mode] for x in R[(R.axis=="axial")&(R.configuration=="nested-tuned")].sort_values("RMSE").itertuples()],
                [128, 58, 63, 58, 55, 65, 55], size=6.8),
          h("Nested-tuned model family comparison: transverse signed angle"),
          table([["Model", "MAE", "MSE", "RMSE", "R2", "Fit ms", "Feature"]] +
                [[x.model, f"{x.MAE:.2f}", f"{x.MSE:.1f}", f"{x.RMSE:.2f}", f"{x.R2:.3f}", f"{x.train_s*1000:.2f}", x.feature_mode] for x in R[(R.axis=="transverse")&(R.configuration=="nested-tuned")].sort_values("RMSE").itertuples()],
                [128, 58, 63, 58, 55, 65, 55], size=6.8), PageBreak()]

story += [h("All requested hyperparameter configurations"),
          p("Each row below is the grouped outer-fold score for that fixed setting, with feature choice still inside training. The family comparison on page 2 uses nested hyperparameter selection, so it is the fair final comparison.", True)]
for axis in ("axial", "transverse"):
    sub = R[(R.axis==axis)&(R.configuration!="nested-tuned")]
    rows = [["Model", "Configuration", "RMSE", "R2"]]
    rows += [[x.model, x.configuration, f"{x.RMSE:.2f}", f"{x.R2:.3f}"] for x in sub.itertuples()]
    story += [h(axis.capitalize()), table(rows, [115, 280, 55, 52], size=6.0)]
story += [PageBreak()]

story += [h("Exploration, sensor response and scaling"), figure("exploration.png"),
          p("The scatter panels plot both FBG readings against angle; the heatmaps show Pearson correlations; boxplots flag extreme wavelength changes. The final panel shows raw, StandardScaler and MinMaxScaler distributions for one input as a visual demo. In actual validation, scalers were fitted only on training folds."),
          p("The axial difference follows signed angle much more clearly than the average. Transverse FBG readings rise with absolute tilt in both directions, so a low signed Pearson correlation does not mean the sensor has no tilt information. Boxplot whiskers are a screening aid, not a reason to drop end angles."), PageBreak()]

story += [h("Model-wise accuracy and training time"), figure("model_comparison.png"),
          p("The four columns show MAE, RMSE, R-squared and mean outer-fold fit time for each axis. These are the nested-tuned family results from page 2. Smaller MAE/RMSE is better; larger R-squared is better. Fit times are too small and environment-dependent to claim a hardware benchmark."),
          p("GPR is strongest on both axes in this run. Degree 3 polynomial improves axial accuracy over degree 2, but the extra flexibility does not reliably solve the transverse sign ambiguity. Greater tree depth and more forest trees do not automatically improve grouped prediction."), PageBreak()]

ax_stat = R[(R.axis=="axial")&(R.configuration=="nested-tuned")&(R.model.isin(["SLR","MLR","Polynomial","Ridge","Lasso","Elastic Net"]))].sort_values("RMSE").iloc[0]
tr_stat = R[(R.axis=="transverse")&(R.configuration=="nested-tuned")&(R.model.isin(["SLR","MLR","Polynomial","Ridge","Lasso","Elastic Net"]))].sort_values("RMSE").iloc[0]
ax_ml = R[(R.axis=="axial")&(R.configuration=="nested-tuned")&(~R.model.isin(["SLR","MLR","Polynomial","Ridge","Lasso","Elastic Net"]))].sort_values("RMSE").iloc[0]
tr_ml = R[(R.axis=="transverse")&(R.configuration=="nested-tuned")&(~R.model.isin(["SLR","MLR","Polynomial","Ridge","Lasso","Elastic Net"]))].sort_values("RMSE").iloc[0]
signed, absolute = A.iloc[0], A.iloc[1]
story += [h("Actual predictions and residuals"), figure("predictions_residuals.png"),
          p(f"The best statistical family is {ax_stat.model} for axial ({ax_stat.RMSE:.2f} deg RMSE) and {tr_stat.model} for signed transverse ({tr_stat.RMSE:.2f} deg). The best conventional ML family is {ax_ml.model} for axial ({ax_ml.RMSE:.2f} deg) and {tr_ml.model} for transverse ({tr_ml.RMSE:.2f} deg). These plots use only outer test predictions."),
          h("Advanced task C: signed versus absolute transverse tilt"),
          p(f"Using the same grouped folds and a simple linear model on average wavelength change, signed angle gives RMSE {signed.RMSE:.2f} deg (R2 {signed.R2:.3f}), while absolute angle gives RMSE {absolute.RMSE:.2f} deg (R2 {absolute.R2:.3f}). That gap is the main thing I learned: the average signal tells me how far the sensor tilted, but generally not which way. Small direction clues in both channels let GPR do better, though I would not assume they stay stable on a new sensor."),
          h("My conclusion"),
          p("For accurate calibration on these 37 readings, I would pick Matérn GPR for axial tilt and also test it for signed transverse tilt. For a simple real-time monitor, a linear differential model is easier to explain and fast for axial angle. I would report transverse magnitude from the average unless direction has been separately calibrated and checked on more runs. The 37-row dataset is tiny, so these errors are evidence for this experiment, not a guarantee in deployment."),
          p('Public code and full row-wise results: <link href="https://github.com/mdminhaj-2106/ML_Lab9" color="blue">https://github.com/mdminhaj-2106/ML_Lab9</link>.', True)]

def footer(canvas, doc):
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#61727d"))
    canvas.drawString(38, 19, "FBG tilt regression | Lab experiment 06")
    canvas.drawRightString(A4[0]-38, 19, f"Page {doc.page}")

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=38, rightMargin=38, topMargin=34, bottomMargin=34)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(PDF)
