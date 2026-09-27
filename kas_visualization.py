"""Plotly-Figuren: Karte für Kaskade/Ausbreitung "in Aktion" (ausgefallene bzw. infizierte/genesene Knoten farbig), G(α)-Kurve mit kritischer Toleranz und Vergleich gegen den einmaligen Ausfall,
β-Endgrößen-Kurve mit Streuband und Pastor-Satorras-Schwelle, Drei-Netze-Vergleichsbalken je Dynamik. Alle Achsen fest (fixedrange), Karten nutzen `scaleanchor` mit autorange und zwei unsichtbaren
Eckpunkten (Hauskonvention)."""

import plotly.graph_objects as go

TEAL, ORANGE, BLUE, RED, PURPLE, GREY, LIGHT, GREEN = "#2F6B65", "#f58518", "#4c78a8", "#e45756", "#7b3fbf", "#b7bec7", "#e8ebee", "#54a24b"


def _lines(xy, pairs):
    xs, ys = [], []
    for u, v in pairs:
        xs += [xy[u][0], xy[v][0], None]
        ys += [xy[u][1], xy[v][1], None]
    return xs, ys


def _corners_trace(xy):
    pad = 0.4
    return go.Scatter(x=[xy[:, 0].min() - pad, xy[:, 0].max() + pad], y=[xy[:, 1].min() - pad, xy[:, 1].max() + pad], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False)


# --- 1 · Kaskade/Ausbreitung in Aktion ---------------------------------------------------------------------------------------------------------------------


def build_cascade_map(xy, edges, failed, giant_nodes):
    """Karte: ausgefallene Knoten/anliegende Kanten hellgrau, überlebende Kanten dünn grau, die größte Restkomponente TEAL hervorgehoben, übrige lebende Knoten (kleinere Komponenten) ORANGE."""
    n = len(xy)
    giant_set = set(giant_nodes)
    alive_edges = [(u, v) for u, v in edges if u not in failed and v not in failed]
    dead_edges = [(u, v) for u, v in edges if u in failed or v in failed]

    fig = go.Figure()
    dx, dy = _lines(xy, dead_edges)
    fig.add_trace(go.Scatter(x=dx, y=dy, mode="lines", line=dict(color=LIGHT, width=1.0), hoverinfo="skip", showlegend=False))
    ax, ay = _lines(xy, alive_edges)
    fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color=GREY, width=1.0), opacity=0.6, hoverinfo="skip", showlegend=False))

    failed_xy = [xy[v] for v in range(n) if v in failed]
    other_xy = [xy[v] for v in range(n) if v not in failed and v not in giant_set]
    giant_xy = [xy[v] for v in range(n) if v in giant_set]

    if failed_xy:
        fig.add_trace(go.Scatter(x=[p[0] for p in failed_xy], y=[p[1] for p in failed_xy], mode="markers", marker=dict(size=6, color=LIGHT, line=dict(width=1, color=GREY)),
                                  name=f"Ausgefallen ({len(failed_xy)})", hoverinfo="skip"))
    if other_xy:
        fig.add_trace(go.Scatter(x=[p[0] for p in other_xy], y=[p[1] for p in other_xy], mode="markers", marker=dict(size=6, color=ORANGE), name=f"Andere Komponente ({len(other_xy)})",
                                  hoverinfo="skip"))
    if giant_xy:
        fig.add_trace(go.Scatter(x=[p[0] for p in giant_xy], y=[p[1] for p in giant_xy], mode="markers", marker=dict(size=7, color=TEAL), name=f"Größte Komponente ({len(giant_xy)})",
                                  hoverinfo="skip"))
    fig.add_trace(_corners_trace(xy))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10), showlegend=True, legend=dict(orientation="h", y=1.08), plot_bgcolor="white")
    fig.update_xaxes(visible=False, fixedrange=True, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, fixedrange=True)
    return fig


def build_sir_map(xy, edges, state):
    """Karte für die Ausbreitung: S (blau), I (rot), R (grün). `state`: Liste 0=S,1=I,2=R je Knoten."""
    n = len(xy)
    ex, ey = _lines(xy, edges)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=GREY, width=1.0), opacity=0.5, hoverinfo="skip", showlegend=False))
    groups = {0: ("Anfällig (S)", BLUE), 1: ("Infiziert (I)", RED), 2: ("Genesen (R)", GREEN)}
    for code, (label, color) in groups.items():
        pts = [xy[v] for v in range(n) if state[v] == code]
        if pts:
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", marker=dict(size=6.5, color=color), name=f"{label} ({len(pts)})", hoverinfo="skip"))
    fig.add_trace(_corners_trace(xy))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10), showlegend=True, legend=dict(orientation="h", y=1.08), plot_bgcolor="white")
    fig.update_xaxes(visible=False, fixedrange=True, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, fixedrange=True)
    return fig


# --- 2 · Kaskadengröße über die Toleranz --------------------------------------------------------------------------------------------------------------------


def build_cascade_alpha_curve(rows, n, critical_tolerance):
    """G(α) - Gesamtausfallanteil über die Toleranz, mit der kritischen Toleranz markiert (kleinstes α ohne Sekundärausfälle) und der Anfangsausfall-Linie (1/n, Stück 8: nur der Startknoten selbst)
    zum Vergleich."""
    alphas = [r["alpha"] for r in rows]
    gs = [r["G"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=alphas, y=gs, mode="lines", line=dict(color=RED, width=2.6), name="G(α) (Kaskade, Gesamtausfall)"))
    fig.add_hline(y=1.0 / n if n else 0.0, line=dict(color=GREY, width=1.4, dash="dot"), annotation_text="Einmaliger Ausfall (Stück 8, nur der Startknoten)", annotation_position="top left",
                  annotation=dict(bgcolor="white"))
    if critical_tolerance is not None:
        fig.add_vline(x=critical_tolerance, line=dict(color=TEAL, width=1.8, dash="dash"), annotation_text=f"kritische Toleranz α*={critical_tolerance:.2f}", annotation_position="top right",
                      annotation=dict(bgcolor="white"))
    top = max(gs) * 1.15 if gs else 1.0
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.16), plot_bgcolor="white")
    fig.update_xaxes(title="Toleranz α", range=[0, max(alphas) * 1.02] if alphas else [0, 1], autorange=False, fixedrange=True)
    fig.update_yaxes(title="G (Gesamtausfallanteil)", range=[0, max(top, 0.05)], autorange=False, fixedrange=True, gridcolor=LIGHT)
    return fig


def build_multiplier_bars(rows):
    """Vervielfachungsfaktor (Kaskadenschaden / einmaliger Ausfall) über die Toleranz - macht sichtbar, um wie viel schlimmer eine Kaskade gegenüber Stück 8 ist."""
    alphas = [r["alpha"] for r in rows]
    mult = [r["multiplier"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{a:.1f}" for a in alphas], y=mult, marker_color=ORANGE))
    fig.add_hline(y=1.0, line=dict(color=GREY, width=1.2, dash="dot"), annotation_text="1× (kein Kaskadeneffekt)", annotation_position="top left")
    top = max(mult) * 1.2 if mult else 1.0
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, plot_bgcolor="white", title=dict(text="Vervielfachungsfaktor gegenüber dem einmaligen Ausfall", x=0.02,
                       font=dict(size=13)))
    fig.update_xaxes(title="Toleranz α", fixedrange=True)
    fig.update_yaxes(range=[0, max(top, 1.5)], autorange=False, fixedrange=True, gridcolor=LIGHT)
    return fig


# --- 3 · Endgröße der Ausbreitung über β ----------------------------------------------------------------------------------------------------------------------


def build_sir_beta_curve(rows, n, beta_c, measured_threshold):
    """Endgröße/n über β mit p10-p90-Streuband über viele Läufe, gemessene Schwelle und Pastor-Satorras-Vorhersage β_c gestrichelt."""
    betas = [r["beta"] for r in rows]
    means = [r["mean"] / n if n else 0.0 for r in rows]
    p10 = [r["p10"] / n if n else 0.0 for r in rows]
    p90 = [r["p90"] / n if n else 0.0 for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=betas + betas[::-1], y=p90 + p10[::-1], fill="toself", fillcolor="rgba(76,120,168,0.18)", line=dict(color="rgba(0,0,0,0)"), name="p10-p90 (Streuband)",
                              hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=betas, y=means, mode="lines", line=dict(color=BLUE, width=2.6), name="mittlere Endgröße/n"))
    if beta_c is not None:
        fig.add_vline(x=beta_c, line=dict(color=RED, width=1.8, dash="dot"), annotation_text=f"Pastor-Satorras β_c={beta_c:.3f}", annotation_position="top right", annotation=dict(bgcolor="white"))
    if measured_threshold is not None:
        fig.add_vline(x=measured_threshold, line=dict(color=TEAL, width=1.8, dash="dash"), annotation_text=f"gemessen β*={measured_threshold:.3f}", annotation_position="top left",
                      annotation=dict(bgcolor="white"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.16), plot_bgcolor="white")
    fig.update_xaxes(title="Ansteckungswahrscheinlichkeit β", range=[0, max(betas) * 1.02] if betas else [0, 1], autorange=False, fixedrange=True)
    fig.update_yaxes(title="mittlere Endgröße / n", range=[0, 1.05], autorange=False, fixedrange=True, gridcolor=LIGHT)
    return fig


# --- 4 · Drei Netze im Vergleich -------------------------------------------------------------------------------------------------------------------------------


def build_network_comparison_bars(rows, metric_key, title, y_title):
    """`rows`: Liste von {"label", metric_key, ...} - ein Balken je Netz, generisch für kritische Toleranz (Kaskade) oder gemessene Epidemieschwelle (Ausbreitung)."""
    labels = [r["label"] for r in rows]
    values = [r[metric_key] if r[metric_key] is not None else 0.0 for r in rows]
    colors = [TEAL, RED, PURPLE]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=colors[: len(labels)], text=[f"{v:.3f}" for v in values], textposition="outside"))
    top = max(values) * 1.3 if values and max(values) > 0 else 1.0
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, plot_bgcolor="white", title=dict(text=title, x=0.02, font=dict(size=13)))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(title=y_title, range=[0, top], autorange=False, fixedrange=True, gridcolor=LIGHT)
    return fig
