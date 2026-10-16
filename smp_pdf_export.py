"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import smp_constants as C
from smp_evaluation import Analysis, de, err_pct, num_text, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "τ": "tau", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "⁻": "-", "⁰": "0", "¹": "1", "²": "2",
            "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Posterior-Sampling"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Gitterpunkte n = {s.n}, Breite der Unschärfe w = {de(s.w, 3)}, Rauschen {de(100 * s.noise, 4)} % des Effektivwerts, Welt: {C.WORLD_SHORT[s.world]}, "
                            f"Prior: {C.PRIOR_SHORT[s.prior]}, Sampler: {C.SAMPLER_SHORT[a.sampler_used]}{' (statt pCN, das einen Gaußschen Prior braucht)' if a.fallback else ''}, "
                            f"{s.chains} Ketten à {s.steps} Schritte, Start: {'im MAP' if s.start == 'map' else 'aus dem Prior'}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Schritt {sci(a.step)}, Annahmerate {de(a.acc, 2)}; R-Dach je Funktional (Mittelwert / Punkt / raue Richtung): " + " / ".join(num_text(v) for v in a.rhats) + ".",
        "Autokorrelationszeit: " + " / ".join(num_text(v) for v in a.iats) + "; effektive Stichprobengröße: " + " / ".join(de(v, 0) for v in a.ess) + ".",
        f"Fehler gegen die Wahrheit: MAP {err_pct(a.err_map)}, Kettenmittel {err_pct(a.err_mean)}; Werte ungleich null: MAP {a.nnz_map}, Mittel {a.nnz_mean} von {s.n}.",
        (f"Referenz ({a.ref_kind}): größter Mittelwertfehler {num_text(a.mean_err_sd)} sd, größter sd-Fehler {de(100 * a.sd_err, 0)} %, sd-Verhältnis Median {de(a.sd_ratio_median, 2)}."
         if a.ref_kind else "Keine Referenz (Laplace-Prior: die Hesse-Matrix des Betrags ist null)."),
        f"Kondition der Matrix: {sci(a.cond)}; wahres Rauschen sigma = {sci(a.sigma)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "MAP", "Kettenmittel", "Ketten-sd"], [16, 26, 30, 34, 38, 36]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(s.n, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), de(a.x_map[i], 3), de(a.mean[i], 3), de(a.sd[i], 4)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if s.n > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {s.n - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Sampler, Diagnostik und Laplace-Näherung für Inverse Probleme. Quelle: posterior-sampling-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
