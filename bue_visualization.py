"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import plotly.graph_objects as go

import bue_constants as C

PALETTE = ["#2F6B65", "#4c78a8", "#c0504d", "#8c6bb1", "#d6732b", "#7f7f7f", "#b59f2a", "#3d9acb"]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _sqrt_line(ns):
    return go.Scatter(x=list(ns), y=[math.sqrt(n) for n in ns], mode="lines", line=dict(color=C.COLOR_SQRT, width=2, dash="dash"), name="Wurzel-Regel √n",
                      hovertemplate="n = %{x}: √n = %{y:.2f}<extra></extra>")


def _log_axes(fig, xtitle="Standorte n", ytitle="Vorteil V = n·SB₁ / SBₙ"):
    fig.update_xaxes(title_text=xtitle, type="log", tickvals=[2, 4, 8, 16, 32, 64, 128, 256], ticktext=["2", "4", "8", "16", "32", "64", "128", "256"])
    fig.update_yaxes(title_text=ytitle, type="log")
    return fig


def build_main(curve: dict, label: str, n_now: int, v_now: float, mc=None):
    """V(n) einer Verteilung (Faltung) gegen die Wurzel-Regel, mit markiertem aktuellem n; `mc` = (n, V, Standardfehler) der Nachprüfung."""
    ns = sorted(curve)
    fig = go.Figure()
    fig.add_trace(_sqrt_line(ns))
    fig.add_trace(go.Scatter(x=ns, y=[curve[n] for n in ns], mode="lines+markers", line=dict(color=C.COLOR_FFT, width=3), name=f"{label} (Faltung)",
                             hovertemplate="n = %{x}: V = %{y:.2f}<extra></extra>"))
    if math.isfinite(v_now):
        fig.add_trace(go.Scatter(x=[n_now], y=[v_now], mode="markers", marker=dict(size=14, color="#e0a800", line=dict(color="black", width=1)), name="aktuelles n",
                                 hovertemplate="n = %{x}: V = %{y:.2f}<extra></extra>"))
    if mc is not None:
        n_mc, v_mc, se = mc
        fig.add_trace(go.Scatter(x=[n_mc], y=[v_mc], mode="markers", marker=dict(size=10, color=C.COLOR_MC, symbol="diamond"),
                                 error_y=dict(type="data", array=[2 * se], visible=True, color=C.COLOR_MC), name="Monte Carlo (± 2 Standardfehler)",
                                 hovertemplate="n = %{x}: V = %{y:.2f}<extra></extra>"))
    return _base(_log_axes(fig))


def build_exp1(rows: list[dict], mc_bands: dict):
    """Experiment 1: V(n) je Verteilung (Linien, Faltung) mit Monte-Carlo-Band über die Seeds der Tabelle (senkrechte Spanne) und die Wurzel-Regel."""
    ns = sorted(rows[0]["v"])
    fig = go.Figure()
    fig.add_trace(_sqrt_line(ns))
    for i, r in enumerate(rows):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=ns, y=[r["v"][n] for n in ns], mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=5), name=r["label"],
                                 legendgroup=r["label"], hovertemplate="n = %{x}: V = %{y:.2f}<extra>" + r["label"] + "</extra>"))
        band = mc_bands.get(r["label"])
        if band:
            xs = sorted(band)
            fig.add_trace(go.Scatter(x=xs, y=[band[n][1] for n in xs], mode="markers", marker=dict(color=color, symbol="diamond-open", size=9),
                                     error_y=dict(type="data", symmetric=False, array=[band[n][2] - band[n][1] for n in xs], arrayminus=[band[n][1] - band[n][0] for n in xs],
                                                  color=color), name=f"{r['label']}, Monte Carlo", legendgroup=r["label"], showlegend=False,
                                     hovertemplate="n = %{x}: Median V = %{y:.2f}<extra>Monte Carlo</extra>"))
    return _base(_log_axes(fig), height=460)


def build_service(rows: list[dict], qs=C.Q_OPTIONS):
    """Experiment 3: V gegen den Servicegrad, eine Linie je Verteilung."""
    fig = go.Figure()
    labels = [f"{str(q).replace('.', ',')} %" for q in qs]
    for i, r in enumerate(rows):
        fig.add_trace(go.Scatter(x=labels, y=[r["v"][q] for q in qs], mode="lines+markers", line=dict(color=PALETTE[i % len(PALETTE)], width=2), name=r["label"],
                                 hovertemplate="Servicegrad %{x}: V = %{y:.2f}<extra>" + r["label"] + "</extra>"))
    fig.update_xaxes(title_text="Servicegrad", type="category")
    fig.update_yaxes(title_text="Vorteil V")
    return _base(fig)


def build_correlation(table: list[dict]):
    """Experiment 4: V gegen Gleichkorrelation, eine Linie je n."""
    fig = go.Figure()
    for i, row in enumerate(table):
        rhos = sorted(row["v"])
        fig.add_trace(go.Scatter(x=rhos, y=[row["v"][r] for r in rhos], mode="lines+markers", line=dict(color=PALETTE[i % len(PALETTE)], width=2), name=f"n = {row['n']}",
                                 hovertemplate="ρ = %{x}: V = %{y:.2f}<extra>n = " + str(row["n"]) + "</extra>"))
    fig.update_xaxes(title_text="Gleichkorrelation ρ")
    fig.update_yaxes(title_text="Vorteil V", type="log")
    return _base(fig)


def build_exponent(rows: list[dict]):
    """Experiment 5: gemessene Steigung (n = 64 bis 256) gegen den asymptotischen Exponenten, ein Punkt je Verteilung und Servicegrad; Diagonale = Übereinstimmung."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.1, 0.7], y=[0.1, 0.7], mode="lines", line=dict(color=C.COLOR_SQRT, dash="dash"), name="gemessen = asymptotisch", hoverinfo="skip"))
    for q, color in zip((0.95, 0.99, 0.999), ("#4c78a8", "#d6732b", "#c0504d")):
        sel = [r for r in rows if abs(r["q"] - q) < 1e-12 and math.isfinite(r["e_high"])]
        fig.add_trace(go.Scatter(x=[r["asymptote"] for r in sel], y=[r["e_high"] for r in sel], mode="markers", marker=dict(size=10, color=color),
                                 text=[r["label"] for r in sel], name=f"Servicegrad {str(q * 100).replace('.0', '').replace('.', ',')} %",
                                 hovertemplate="%{text}<br>asymptotisch %{x:.3f}, gemessen %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="asymptotischer Exponent", range=[0.1, 0.7])
    fig.update_yaxes(title_text="gemessene Steigung (n = 64 bis 256)", range=[0.1, 0.7])
    return _base(fig)
