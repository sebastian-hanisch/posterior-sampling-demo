"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import smp_constants as C
import smp_model as model
from smp_evaluation import Analysis, de, iat_text, noise_label, num_text, step_text


def _ref_line(fig, value, text, position, axis, log=True, **line_kwargs):
    """Referenzlinie mit Beschriftung auf einer logarithmischen Achse: die Linie liegt auf dem Datenwert, die Beschriftung braucht log10 des Werts
    (add_hline und add_vline setzen beide auf denselben Wert, die Beschriftung sitzt dann falsch und dehnt die Achse); log=False für lineare Achsen."""
    if log and not value > 0.0:
        return
    pos = math.log10(value) if log else value
    top, left = "top" in position, "left" in position
    if axis == "y":
        fig.add_hline(y=value, **line_kwargs)
        if text:
            fig.add_annotation(xref="paper", yref="y", x=0 if left else 1, y=pos, text=text, showarrow=False, xanchor="left" if left else "right", yanchor="bottom" if top else "top")
    else:
        fig.add_vline(x=value, **line_kwargs)
        if text:
            fig.add_annotation(xref="x", yref="paper", x=pos, y=1 if top else 0, text=text, showarrow=False, xanchor="right" if left else "left", yanchor="top" if top else "bottom")


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    data = {"Punkt": [f"{i + 1}" for i in range(k)], "Ort s": a.s[:k], "Wahrheit x": a.x[:k], "MAP": a.x_map[:k], "Kettenmittel": a.mean[:k], "Ketten-sd": a.sd[:k]}
    if a.ref_mean is not None:
        data[f"Referenz-Mittel ({a.ref_kind})"] = a.ref_mean[:k]
        data[f"Referenz-sd ({a.ref_kind})"] = a.ref_sd[:k]
    return pd.DataFrame(data)


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 4)) for c in df.columns if c != "Punkt"})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, MAP, Kettenmittel mit 95-%-Band aus den Ketten und die Referenz (exakter Gauß-Posterior oder Laplace-Näherung); die Achse ist fest."""
    lo, hi = C.SHOW_RANGE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="95-%-Intervall der Ketten", hoverinfo="skip"))
    if a.ref_mean is not None:
        fig.add_trace(go.Scatter(x=a.s, y=a.ref_mean + C.Z_BAND * a.ref_sd, mode="lines", name=f"Referenz ({a.ref_kind}): ±1,96 sd", line=dict(color=C.COLOR_EXACT, width=1, dash="dot"),
                                 legendgroup="ref", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=a.s, y=a.ref_mean - C.Z_BAND * a.ref_sd, mode="lines", line=dict(color=C.COLOR_EXACT, width=1, dash="dot"), legendgroup="ref", showlegend=False,
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_map, mode="lines", name="MAP", line=dict(color=C.COLOR_MAP, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.mean, mode="lines+markers", name="Kettenmittel", line=dict(color=C.COLOR_MEAN, width=2.5), marker=dict(size=5),
                             hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    return _fixed(fig, 400)


def build_trace_chart(a: Analysis, f: int) -> go.Figure:
    """Spur eines Funktionals (ausgedünnt) für alle Ketten; die Referenz (exakt oder Laplace) als Linie mit ±2 sd."""
    fig = go.Figure()
    for c in range(a.traces.shape[1]):
        fig.add_trace(go.Scatter(x=a.trace_steps, y=a.traces[:, c, f], mode="lines", name=f"Kette {c + 1}", line=dict(color=C.COLOR_CHAINS[c % len(C.COLOR_CHAINS)], width=1)))
    if a.ref_mean is not None:
        fmat = model.functionals(a.s)
        ref = float(fmat[f] @ a.ref_mean)
        fig.add_hline(y=ref, line=dict(color=C.COLOR_EXACT, width=2, dash="dash"), annotation_text=f"Referenz ({a.ref_kind})", annotation_position="top left")
    fig.add_vline(x=a.burn, line=dict(color=C.COLOR_GREY, width=1, dash="dot"), annotation_text="verworfen bis hier", annotation_position="top right")
    fig.update_xaxes(title="Schritt")
    fig.update_yaxes(title=model.FUNCTIONAL_NAMES[f])
    return _fixed(fig, 360)


def build_sd_chart(a: Analysis) -> go.Figure:
    """Standardabweichung aus den Ketten gegen die Referenz je Gitterpunkt."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.sd, mode="lines+markers", name="Ketten", line=dict(color=C.COLOR_MEAN, width=3), marker=dict(size=5)))
    if a.ref_sd is not None:
        fig.add_trace(go.Scatter(x=a.s, y=a.ref_sd, mode="lines", name=f"Referenz ({a.ref_kind})", line=dict(color=C.COLOR_EXACT, width=2, dash="dash")))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Standardabweichung", type="log")
    return _fixed(fig, 340)


def build_sampler_info_chart(res: dict) -> go.Figure:
    """Autokorrelationszeit des Punktwerts je Rauschstufe und Sampler (logarithmisch)."""
    fig = go.Figure()
    xs = [noise_label(n) for n in C.EXP_NOISE]
    for kind in C.SAMPLERS:
        ys = [next(r for r in res["info"] if r["noise"] == nz and r["sampler"] == kind)["iats"][1] for nz in C.EXP_NOISE]
        fig.add_trace(go.Bar(x=xs, y=[min(v, 1e5) for v in ys], name=C.SAMPLER_SHORT[kind], marker_color=C.SAMPLER_COLORS[kind], text=[iat_text(v) for v in ys], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Rauschen (% des Effektivwerts der Daten)", type="category")
    fig.update_yaxes(title="Autokorrelationszeit des Punktwerts (Schritte)", type="log", range=[1, 4])
    return _fixed(fig, 360)


def build_dimension_chart(res: dict) -> go.Figure:
    """Annahmerate mit dem bei n = 24 abgestimmten Schritt über die Dimension."""
    dims = res["dims"]
    xs = [d["n"] for d in dims]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[d["fixed_pcn_acc"] for d in dims], mode="lines+markers", name="pCN", line=dict(color=C.SAMPLER_COLORS["pcn"], width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=xs, y=[d["fixed_rwm_acc"] for d in dims], mode="lines+markers", name="RWM isotrop", line=dict(color=C.SAMPLER_COLORS["rwm_iso"], width=3), marker=dict(size=8)))
    fig.update_xaxes(title="Zahl der Unbekannten n (Messwerte fest bei 24)", type="log", tickvals=xs, ticktext=[str(v) for v in xs])
    fig.update_yaxes(title="Annahmerate mit dem Schritt von n = 24", range=[0, 0.6])
    return _fixed(fig, 340)


def build_burnin_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    for key, label, color in (("map", "Start um das MAP", C.COLOR_MEAN), ("prior", "Start aus dem Prior", C.COLOR_MAP)):
        rows = res["rows"][key]
        fig.add_trace(go.Scatter(x=[r["discard"] for r in rows], y=[r["rhat_point"] for r in rows], mode="lines+markers", name=f"{label}: Punktwert", line=dict(color=color, width=3)))
        fig.add_trace(go.Scatter(x=[r["discard"] for r in rows], y=[r["rhat_mean"] for r in rows], mode="lines+markers", name=f"{label}: Mittelwert", line=dict(color=color, width=2, dash="dash")))
    _ref_line(fig, C.RHAT_OK, "R̂ = 1,1", "bottom right", "y", line=dict(color=C.COLOR_GREY, dash="dot"))
    fig.update_xaxes(title="Verworfene Schritte")
    fig.update_yaxes(title="R̂ (Gelman-Rubin)", type="log")
    return _fixed(fig, 340)


def build_identification_chart(res: dict) -> go.Figure:
    xs = [noise_label(n) for n in res["noise"]]
    fig = go.Figure()
    for row, color in zip(res["rows"], (C.COLOR_MEAN, C.COLOR_MAP, C.COLOR_GREY)):
        fig.add_trace(go.Bar(x=xs, y=row["counts"], name=row["label"], marker_color=color, text=row["counts"], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Rauschen (% des Effektivwerts der Daten)", type="category")
    fig.update_yaxes(title=f"Richtungen mit λ > 1 (von {res['n']})", range=[0, res["n"] + 6])
    return _fixed(fig, 340)


def build_nongauss_chart(res: dict) -> go.Figure:
    labels = [f"{noise_label(r['noise'])}, b = {de(r['scale'], 2)}" for r in res["laplace"]]
    fig = go.Figure()
    for key, name, color in (("map", "MAP (= L1)", C.COLOR_MAP), ("mean", "Posterior-Mittel", C.COLOR_MEAN), ("tik", "Tikhonov, bester λ", C.COLOR_GREY)):
        fig.add_trace(go.Bar(x=labels, y=[100.0 * r[key] for r in res["laplace"]], name=name, marker_color=color, text=[f"{100 * r[key]:.1f} %" for r in res["laplace"]], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Rauschen und Skala des Laplace-Priors", type="category")
    fig.update_yaxes(title="Fehler (% der Signalnorm)", range=[0, 6])
    return _fixed(fig, 340)


def build_laplace_chart(res: dict) -> go.Figure:
    labels = [f"ε {de(r['eps'], 3)}, b {de(r['scale'], 2)}, {noise_label(r['noise'])}" for r in res["huber"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["ratio_median"] for r in res["huber"]], name="Median", marker_color=C.COLOR_LAPLACE,
                         error_y=dict(type="data", symmetric=False, array=[r["ratio_max"] - r["ratio_median"] for r in res["huber"]], arrayminus=[r["ratio_median"] - r["ratio_min"] for r in res["huber"]],
                                      color="#555555"),
                         text=[de(r["ratio_median"], 2) for r in res["huber"]], textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text="Laplace-sd = Ketten-sd", annotation_position="top left")
    fig.update_xaxes(title="Pseudo-Huber-Prior: Glättung, Skala und Rauschen", type="category")
    fig.update_yaxes(title="sd der Laplace-Näherung / sd der Ketten (Balken: min bis max)", range=[0, 1.7])
    return _fixed(fig, 360, showlegend=False)
