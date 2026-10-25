"""src/academic_reporter.py - AcademicReporter: Excel(10 sheets), PDF, DOCX, LaTeX, HTML."""
from __future__ import annotations
import datetime, hashlib, io, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, Optional
from config import CONFIG, BiogasConfig

VERSION = "v11.0-Omniscient"
_TS = lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
_SHA256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


class AcademicReporter:
    """Penta-format academic reporting engine."""

    def __init__(self, results: dict, cfg: BiogasConfig = CONFIG):
        self.results = results
        self.cfg = cfg

    def _df(self, key, max_rows: int = 1000) -> pd.DataFrame:
        data = self.results.get(key)
        if data is None: return pd.DataFrame({"info": [f"{key}: not computed"]})
        if isinstance(data, dict): return pd.DataFrame([data])
        if isinstance(data, pd.Series): df = data.reset_index(); df.columns=["Parameter","Value"]; return df
        if isinstance(data, pd.DataFrame): return data.head(max_rows)
        return pd.DataFrame({"value":[str(data)]})

    # =========================================================================
    # EXCEL — 10 sheets
    # =========================================================================
    def export_excel(self) -> Path:
        try: import openpyxl; from openpyxl.styles import Font, PatternFill, Alignment
        except ImportError: raise ImportError("pip install openpyxl")
        wb = openpyxl.Workbook(); wb.remove(wb.active)
        hdr_fill = PatternFill("solid", fgColor="1F4E79")
        hdr_font = Font(bold=True, color="FFFFFF", size=10)
        alt_fill = PatternFill("solid", fgColor="D6E4F0")

        def add_sheet(name, df):
            ws = wb.create_sheet(name[:31])
            for ci, col in enumerate(df.columns, 1):
                c = ws.cell(row=1, column=ci, value=str(col))
                c.fill = hdr_fill; c.font = hdr_font; c.alignment = Alignment(horizontal="center")
            for ri, row in enumerate(df.itertuples(index=False), 2):
                f = alt_fill if ri % 2 == 0 else PatternFill()
                for ci, val in enumerate(row, 1):
                    cell = ws.cell(row=ri, column=ci,
                        value=round(val,4) if isinstance(val,float) and not isinstance(val,bool) else val)
                    cell.fill = f
            for col in ws.columns: ws.column_dimensions[col[0].column_letter].width = 18

        sheets = [
            ("Raw Empirical", "empirical_raw"),
            ("Raw Synthetic", "synthetic_raw"),
            ("Viterbi States", "viterbi_states"),
            ("Model Comparison", "omniscient_comparison"),
            ("K-Sensitivity", "k_sensitivity"),
            ("HMM TransMats", "hmm_transmat"),
            ("EVT Tail Compare", "evt_comparison"),
            ("Hysteresis Log", "hysteresis_log"),
            ("OU Calibration", "ou_calibration"),
            ("MCDA Ranking", "mcda_ranking"),
        ]
        for name, key in sheets:
            add_sheet(name, self._df(key))

        out = self.cfg.output_path(self.cfg.REPORT_EXCEL)
        wb.save(str(out)); return out

    # =========================================================================
    # PDF — 6 sections + 3 appendices (fpdf2)
    # =========================================================================
    def export_pdf(self) -> Path:
        try: from fpdf import FPDF
        except ImportError: raise ImportError("pip install fpdf2")
        pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        # Title page
        pdf.set_font("Helvetica","B",18)
        pdf.set_fill_color(31,78,121); pdf.set_text_color(255,255,255)
        pdf.cell(0,14,f"BIOGAS RS-PIM Academic Methodology Report {VERSION}",ln=True,fill=True,align="C")
        pdf.set_text_color(0,0,0)
        pdf.set_font("Helvetica","",10)
        pdf.cell(0,7,f"Generated: {_TS()} | Q1 Journal Supplementary Material",ln=True,align="C")
        pdf.ln(5)
        # Journal badge
        pdf.set_font("Helvetica","I",9)
        pdf.multi_cell(0,6,"Target venues: ASTIN Bulletin | Insurance: Mathematics & Economics | "
                       "European Journal of Operational Research | JRSS")
        pdf.ln(4)

        def section(title):
            pdf.add_page()
            pdf.set_fill_color(31,78,121); pdf.set_text_color(255,255,255)
            pdf.set_font("Helvetica","B",13)
            pdf.cell(0,9,title,ln=True,fill=True)
            pdf.set_text_color(0,0,0); pdf.set_font("Helvetica","",9); pdf.ln(2)

        def table(df, max_rows=20):
            df2=df.head(max_rows); cols=list(df2.columns)[:7]; w=180/max(len(cols),1)
            pdf.set_font("Helvetica","B",8); pdf.set_fill_color(200,220,240)
            for c in cols: pdf.cell(w,6,str(c)[:20],border=1,fill=True)
            pdf.ln()
            pdf.set_font("Helvetica","",7)
            for _, row in df2.iterrows():
                for c in cols:
                    v=row[c]; txt=f"{v:.3f}" if isinstance(v,float) else str(v)[:20]
                    pdf.cell(w,5,txt,border=1)
                pdf.ln()
            pdf.ln(2)

        def para(text):
            pdf.set_font("Helvetica","",9); pdf.multi_cell(0,5,text); pdf.ln(2)

        # Section 1: OU SDE
        section("Section 1: OU SDE Mathematical Derivation")
        para("The Ornstein-Uhlenbeck SDE: dX_t = theta*(mu - X_t)dt + sigma*dW_t")
        para(f"Calibrated: theta={CONFIG.OU_THETA}, mu={CONFIG.OU_MU}, sigma={CONFIG.OU_SIGMA}, dt={CONFIG.OU_DT}")
        para("Exact transition moments: E[X_{t+dt}|X_t] = mu + (X_t-mu)*exp(-theta*dt)")
        para("Var[X_{t+dt}|X_t] = sigma^2/(2*theta)*(1-exp(-2*theta*dt))")
        para("Kalman-RTS smoother provides minimum-variance state estimates without distributional assumptions on process noise.")
        table(self._df("ou_calibration"))

        # Section 2: Model Comparison
        section("Section 2: Omniscient 13-Model Comparison")
        para("13 model families fitted for k in {2,...,8}. 12 criteria computed per fit.")
        table(self._df("omniscient_comparison"))

        # Section 3: K-Sensitivity
        section("Section 3: K-Sensitivity Analysis (BIC/ICL/CAIC/HQIC Elbow)")
        table(self._df("k_sensitivity"))

        # Section 4: Hysteresis
        section("Section 4: Biological Hysteresis & Monod Kinetics")
        para("Hysteresis kernel: IF S_{t-1}=Worst AND S_t=Best AND pH<6.8 THEN S_t <- k-2")
        para("Monod inhibition: I(pH) = exp(-alpha*(pH_crit-pH)^2) for pH < pH_crit=6.5")
        para("Thermodynamic constraint: delta_G_acetoclastic < 0 for methanogenesis feasibility")
        table(self._df("hysteresis_log"))

        # Section 5: EVT
        section("Section 5: EVT Tail Comparison (5 Distributions)")
        para("Pickands-Balkema-de Haan theorem guarantees GPD convergence for excess distribution above threshold.")
        para("5 competing distributions: GPD, Lognormal, Gamma, Weibull, Burr(XII)")
        para("Goodness-of-fit: KS, Anderson-Darling, Cramer-von Mises; AIC/BIC model selection")
        table(self._df("evt_comparison"))

        # Section 6: MCDA
        section("Section 6: MCDA Weighted Scoring")
        para(f"Weights: BIC={CONFIG.MCDA_WEIGHT_BIC}, ICL={CONFIG.MCDA_WEIGHT_ICL}, "
             f"Sil={CONFIG.MCDA_WEIGHT_SILHOUETTE}, PredLL={CONFIG.MCDA_WEIGHT_PREDICTIVE_LL}, "
             f"Conv={CONFIG.MCDA_WEIGHT_CONVERGENCE}")
        table(self._df("mcda_ranking"))

        # Appendices
        section("Appendix A: HMM Transition Matrices (k=2..8)")
        table(self._df("hmm_transmat"))

        section("Appendix B: Complete Parameter Tables")
        table(self._df("omniscient_comparison"), max_rows=50)

        section("Appendix C: Bootstrap CIs for VaR/TVaR")
        table(self._df("bootstrap_ci"))

        out = self.cfg.output_path(self.cfg.REPORT_PDF)
        pdf.output(str(out)); return out

    # =========================================================================
    # DOCX — executive summary (python-docx)
    # =========================================================================
    def export_docx(self) -> Path:
        try: from docx import Document; from docx.shared import Pt, RGBColor, Inches
        except ImportError: raise ImportError("pip install python-docx")
        doc = Document()
        doc.core_properties.author = "BIOGAS RS-PIM Suite"
        doc.core_properties.subject = "Q1 Journal Supplementary Material"

        h = doc.add_heading(f"BIOGAS RS-PIM Executive Summary {VERSION}", 0)
        doc.add_paragraph(f"Generated: {_TS()}")
        doc.add_paragraph("Prepared for: Dissertation Defense Committee / Corporate Underwriting Board")

        def add_table_sec(title, df, max_rows=20):
            doc.add_heading(title, level=2)
            df2 = df.head(max_rows)
            cols = list(df2.columns)[:8]
            t = doc.add_table(rows=1, cols=len(cols)); t.style = "Light List Accent 1"
            for i, c in enumerate(cols): t.rows[0].cells[i].text = str(c)
            for _, row in df2.iterrows():
                cells = t.add_row().cells
                for i, c in enumerate(cols):
                    v = row[c]; cells[i].text = f"{v:.3f}" if isinstance(v,float) else str(v)

        doc.add_heading("1. Optimal Model Selection (MCDA)", level=1)
        solvency = self.results.get("solvency_metrics", {})
        doc.add_paragraph(f"SCR (VaR 99.5%): {solvency.get('SCR', 'N/A')}")
        doc.add_paragraph(f"Commercial Premium: {solvency.get('solvency_premium', 'N/A')}")
        doc.add_paragraph(f"TVaR 99.5%: {solvency.get('TVaR_99.5%', 'N/A')}")
        doc.add_paragraph(f"EPD Ratio: {solvency.get('EPD_ratio', 'N/A')}")

        for title, key in [("MCDA Ranking", "mcda_ranking"),
                            ("EVT Distribution Comparison", "evt_comparison"),
                            ("Solvency II Metrics", "solvency_metrics"),
                            ("OU Calibration", "ou_calibration")]:
            add_table_sec(title, self._df(key))

        doc.add_heading("Risk Capital Calculation", level=2)
        doc.add_paragraph("Risk Margin = Pure Expected Loss + 6% x (TVaR - Pure Expected Loss)")
        doc.add_paragraph("(EIOPA cost-of-capital rate: 6%; Solvency II Directive Article 101)")

        out = self.cfg.output_path(self.cfg.REPORT_DOCX)
        doc.save(str(out)); return out

    # =========================================================================
    # LATEX — standalone .tex for Overleaf
    # =========================================================================
    def export_latex(self) -> Path:
        def df_to_booktabs(df, caption, label, max_rows=20):
            df2 = df.head(max_rows); cols = list(df2.columns)[:7]
            spec = "l" + "r"*(len(cols)-1)
            hdrs = " & ".join(rf"\textbf{{{str(c)[:14]}}}" for c in cols)
            lines = [rf"\begin{{table}}[H]\centering\small",
                     rf"\caption{{{caption}}}\label{{{label}}}",
                     rf"\begin{{tabular}}{{{spec}}}\toprule",
                     hdrs + r" \\\midrule"]
            for _, row in df2.iterrows():
                cells = []
                for c in cols:
                    v = row[c]
                    cells.append(f"{v:.3f}" if isinstance(v,float) else str(v)[:16].replace("_", r"\_"))
                lines.append(" & ".join(cells) + " \\\\")
            lines += [r"\bottomrule\end{tabular}", r"\end{table}", ""]
            return "\n".join(lines)

        tex_lines = [
            r"\documentclass[12pt,a4paper]{article}",
            r"\usepackage{booktabs,float,geometry,amsmath,amssymb,lmodern}",
            r"\geometry{margin=2.5cm}",
            rf"\title{{BIOGAS RS-PIM Academic Methodology Appendix {VERSION}}}",
            rf"\date{{{_TS()}}}",
            r"\begin{document}\maketitle\tableofcontents\newpage",
            "",
            r"\section{OU SDE Theoretical Framework}",
            "The Ornstein-Uhlenbeck process satisfies:",
            r"\begin{equation}dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t\end{equation}",
            rf"Calibrated parameters: $\theta={CONFIG.OU_THETA}$, $\mu={CONFIG.OU_MU}$, $\sigma={CONFIG.OU_SIGMA}$, $\Delta t={CONFIG.OU_DT}$.",
            "Exact transition moments:",
            r"\begin{align}",
            r"\mathbb{E}[X_{t+\Delta t}|X_t] &= \mu + (X_t-\mu)e^{-\theta\Delta t}\\",
            r"\mathrm{Var}[X_{t+\Delta t}|X_t] &= \frac{\sigma^2}{2\theta}(1-e^{-2\theta\Delta t})",
            r"\end{align}",
            df_to_booktabs(self._df("ou_calibration"), "OU SDE Calibration", "tab:ou"),
            "",
            r"\section{Omniscient 13-Model Comparison}",
            r"All 13 model families fitted for $k \in \{2,3,4,5,6,7,8\}$.",
            df_to_booktabs(self._df("omniscient_comparison"), "Omniscient Model Comparison (MCDA Ranked)", "tab:omni"),
            "",
            r"\section{K-Sensitivity Analysis}",
            "Information criteria: BIC, ICL, AICc, CAIC, HQIC for all $k$.",
            df_to_booktabs(self._df("k_sensitivity"), "K-Value Sensitivity Analysis", "tab:ksens"),
            "",
            r"\section{Biological Hysteresis: Monod Kinetics}",
            r"Hysteresis override: $S_{t|\mathrm{Worst}\to\mathrm{Best}, \mathrm{pH}<6.8} \leftarrow k-2$",
            r"\begin{equation}I(\mathrm{pH}) = \exp\left(-\alpha(\mathrm{pH}_{\mathrm{crit}}-\mathrm{pH})^2\right), \quad \mathrm{pH}<\mathrm{pH}_{\mathrm{crit}}\end{equation}",
            df_to_booktabs(self._df("hysteresis_log"), "Hysteresis Correction Log", "tab:hyst"),
            "",
            r"\section{EVT Tail Model Comparison (5 Distributions)}",
            "POT-GPD (Pickands-Balkema-de Haan theorem) vs.~Lognormal, Gamma, Weibull, Burr~XII.",
            r"\begin{align}",
            r"\mathrm{VaR}_p &= u + \frac{\hat{\sigma}}{\hat{\xi}}\left[\left(\frac{n(1-p)}{n_u}\right)^{-\hat{\xi}}-1\right]\\",
            r"\mathrm{TVaR}_p &= \frac{\mathrm{VaR}_p}{1-\hat{\xi}} + \frac{\hat{\sigma}-\hat{\xi}u}{1-\hat{\xi}}",
            r"\end{align}",
            df_to_booktabs(self._df("evt_comparison"), "EVT 5-Distribution Goodness-of-Fit", "tab:evt"),
            "",
            r"\section{MCDA Weighted Scoring}",
            rf"Weights: $w_{{\mathrm{{BIC}}}}={CONFIG.MCDA_WEIGHT_BIC}$, $w_{{\mathrm{{ICL}}}}={CONFIG.MCDA_WEIGHT_ICL}$, "
            rf"$w_{{\mathrm{{Sil}}}}={CONFIG.MCDA_WEIGHT_SILHOUETTE}$, "
            rf"$w_{{\mathrm{{LL}}}}={CONFIG.MCDA_WEIGHT_PREDICTIVE_LL}$, $w_{{\mathrm{{Conv}}}}={CONFIG.MCDA_WEIGHT_CONVERGENCE}$.",
            df_to_booktabs(self._df("mcda_ranking"), "MCDA Weighted Ranking", "tab:mcda"),
            "",
            r"\appendix",
            r"\section{Bootstrap Confidence Intervals for VaR/TVaR}",
            df_to_booktabs(self._df("bootstrap_ci"), "Bootstrap CI (B=2000)", "tab:bsci"),
            "",
            r"\end{document}",
        ]
        out = self.cfg.output_path(self.cfg.REPORT_LATEX)
        out.write_text("\n".join(tex_lines), encoding="utf-8")
        return out

    # =========================================================================
    # HTML — self-contained with embedded Plotly
    # =========================================================================
    def export_html(self) -> Path:
        try:
            import plotly.express as px
            import plotly.io as pio
            HAS_PLOTLY = True
        except ImportError:
            HAS_PLOTLY = False

        def df_to_html(df, max_rows=30, collapsible=True) -> str:
            df2 = df.head(max_rows)
            hdrs = "".join(f"<th>{c}</th>" for c in df2.columns)
            rows_html = ""
            for _, row in df2.iterrows():
                cells = ""
                for v in row:
                    txt = f"{v:.4f}" if isinstance(v,float) else str(v)[:40]
                    cells += f"<td>{txt}</td>"
                rows_html += f"<tr>{cells}</tr>"
            tbl = f'<table class="rtable"><thead><tr>{hdrs}</tr></thead><tbody>{rows_html}</tbody></table>'
            if collapsible:
                uid = abs(hash(hdrs)) % 100000
                return (f'<details><summary>Show/Hide Table ({min(len(df),max_rows)} rows)</summary>'
                        f'{tbl}</details>')
            return tbl

        chart_html = ""
        if HAS_PLOTLY:
            omni = self.results.get("omniscient_comparison")
            if isinstance(omni, pd.DataFrame) and "BIC" in omni.columns and "k" in omni.columns:
                plot_df = omni.dropna(subset=["BIC","k"])
                plot_df = plot_df[plot_df["k"].between(2,8)]
                fig = px.scatter(plot_df, x="k", y="BIC", color="model", symbol="model",
                    title="BIC vs k — All 13 Models", template="plotly_white", width=950, height=480)
                chart_html += pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            mcda = self.results.get("mcda_ranking")
            if isinstance(mcda, pd.DataFrame) and "MCDA_Score" in mcda.columns:
                top20 = mcda.dropna(subset=["MCDA_Score"]).head(20)
                labels = [f"{r['model']}(k={r['k']})" for _, r in top20.iterrows()]
                fig2 = px.bar(top20, x="MCDA_Score", y=labels, orientation="h",
                    title="MCDA Score — Top 20 Configurations", template="plotly_white",
                    width=950, height=500)
                chart_html += pio.to_html(fig2, full_html=False, include_plotlyjs=False)

        sections_html = ""
        for key, title in [("omniscient_comparison","Omniscient Model Comparison"),
                            ("k_sensitivity","K-Sensitivity Analysis"),
                            ("mcda_ranking","MCDA Ranking"),
                            ("evt_comparison","EVT Distribution Comparison"),
                            ("solvency_metrics","Solvency II Metrics"),
                            ("ou_calibration","OU SDE Calibration"),
                            ("hysteresis_log","Hysteresis Correction Log"),
                            ("bootstrap_ci","Bootstrap CIs (VaR/TVaR)")]:
            sections_html += f"<h2>{title}</h2>\n{df_to_html(self._df(key))}\n"

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>BIOGAS RS-PIM {VERSION} Report</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:1280px;margin:auto;padding:20px;background:#f0f4f8}}
  h1{{color:#1F4E79;border-bottom:4px solid #1F4E79;padding-bottom:8px}}
  h2{{color:#2E75B6;border-bottom:2px solid #2E75B6;margin-top:28px}}
  .badge{{background:#1F4E79;color:white;padding:4px 12px;border-radius:4px;font-size:12px;display:inline-block;margin-bottom:16px}}
  .rtable{{border-collapse:collapse;width:100%;background:white;margin:8px 0;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
  .rtable th{{background:#1F4E79;color:white;padding:6px 10px;text-align:left;font-size:11px;white-space:nowrap}}
  .rtable td{{padding:4px 10px;border:1px solid #ddd;font-size:11px}}
  .rtable tr:nth-child(even){{background:#EBF5FF}}
  details{{background:white;border:1px solid #ccd;border-radius:4px;padding:8px;margin:6px 0}}
  summary{{cursor:pointer;font-weight:bold;color:#2E75B6}}
  @media print{{.rtable{{font-size:9px}}}}
</style></head>
<body>
<h1>BIOGAS RS-PIM Academic Report {VERSION}</h1>
<span class="badge">Q1 Journal Supplementary Material</span>
<p>Generated: {_TS()}</p>
{chart_html}
{sections_html}
<hr>
<p style="font-size:10px;color:#888">Generated by BIOGAS RS-PIM Suite {VERSION} | Ph.D. Dissertation Research</p>
</body></html>"""

        out = self.cfg.output_path(self.cfg.REPORT_HTML)
        out.write_text(html, encoding="utf-8")
        return out

    # =========================================================================
    # EXPORT ALL + SHA-256 checksums
    # =========================================================================
    def export_all(self) -> Dict[str, str]:
        """Run all 5 formats. Return dict of format -> path/error + SHA-256."""
        out = {}
        for fmt, fn in [("Excel", self.export_excel), ("PDF", self.export_pdf),
                        ("DOCX", self.export_docx), ("LaTeX", self.export_latex),
                        ("HTML", self.export_html)]:
            try:
                p = fn()
                sha = _SHA256(p)
                out[fmt] = {"path": str(p), "sha256_prefix": sha, "status": "OK"}
            except Exception as e:
                out[fmt] = {"path": None, "status": f"ERROR: {e}"}
        return out
