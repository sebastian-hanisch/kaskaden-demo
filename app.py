"""Kaskaden und Ausbreitung – lastbasierte Kaskadenausfälle, SIR-Epidemieschwelle – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Graphen-und-Netzwerke-Reihe, Kind der Robustheit (Stück 8): dort war jeder Knotenausfall ein einmaliges, unabhängiges Ereignis - hier geht es um FOLGEEREIGNISSE, die aus einem
einzigen Ausfall eine viel größere Katastrophe machen. Zwei Dynamiken: lastbasierte Kaskadenausfälle (Motter & Lai 2002) und SIR-Ausbreitung (Pastor-Satorras & Vespignani 2001).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kas_algorithm as A
import kas_constants as C
import kas_evaluation as ev
import kas_visualization as viz
from kas_evaluation import measured_threshold
from kas_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, store_from_widget, sync_query_params, push_to_widget

st.set_page_config(page_title="Kaskaden und Ausbreitung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _cascade_analysis(settings):
    return ev.analyse_cascade(settings)


@st.cache_data(show_spinner=False)
def _sir_analysis(settings):
    return ev.analyse_sir(settings)


@st.cache_data(show_spinner=False)
def _cascade_alpha_sweep(settings):
    return ev.cascade_alpha_sweep(settings)


@st.cache_data(show_spinner=False)
def _sir_beta_sweep(settings, betas):
    return ev.sir_beta_sweep(settings, betas)


@st.cache_data(show_spinner=False)
def _network_comparison(city_settings, ba_settings, barbell_settings):
    return ev.network_comparison(city_settings, ba_settings, barbell_settings)


def _german(x):
    return f"{x:,}".replace(",", ".") if isinstance(x, int) else x


BETA_SWEEP_DEFAULT = tuple(round(0.01 * i, 3) for i in range(1, 61))

st.title("🌊 Kaskaden und Ausbreitung")
st.markdown(
    """
**Neuntes Stück der Graphen-und-Netzwerke-Reihe**, Kind der Robustheit (Stück 8: dort war jeder Knotenausfall ein EINMALIGES, unabhängiges Ereignis - hier geht es um FOLGEEREIGNISSE). Zwei
Dynamiken: **Lastbasierte Kaskadenausfälle** (Motter & Lai 2002, Phys. Rev. E 66, 065102(R)): jeder Knoten trägt eine Anfangslast (seine Betweenness) und eine Kapazität = (1+α)·Anfangslast
(Toleranz α); fällt ein Knoten aus, verteilen sich die kürzesten Wege neu - manche Knoten tragen jetzt mehr Last als ihre Kapazität und fallen ebenfalls aus, eine Kettenreaktion, die den einmaligen
Schaden aus Stück 8 um ein Vielfaches übertreffen kann. **SIR-Ausbreitung** (Pastor-Satorras & Vespignani 2001, Phys. Rev. Lett. 86(14), 3200-3203): eine stochastische Infektion (S→I→R, diskrete
Zeit) verbreitet sich über die Kanten - skalenfreie Netze haben eine deutlich niedrigere Epidemieschwelle als homogene Netze (hier an den drei Vehikeln GEMESSEN, nicht vorausgesetzt).
"""
)
st.caption(
    "Kind der Robustheits-Demo (achtes Stück der Graphen-und-Netzwerke-Reihe). Der Barbell eignet sich für BEIDE Dynamiken als von-Hand-Beispiel: fällt der Brückenknoten aus, ist der Graph "
    "sofort in zwei Teile getrennt - für die Kaskade heißt das: keine Lastumverteilung über die (nicht mehr existierende) Brücke mehr möglich, die Kaskade endet augenblicklich; für SIR heißt das: "
    "eine Infektion in der einen Clique kann die andere nur über den Brückenknoten selbst erreichen."
)

with st.expander("So funktioniert die Messung", expanded=True):
    st.markdown(
        """
1. **Kaskade (Motter & Lai 2002):** Anfangslast = Betweenness auf dem VOLLEN Graphen (einmal berechnet), Kapazität = (1+α)·Anfangslast. Rundenweise: Betweenness auf dem Restgraphen neu berechnen,
   jeder Knoten mit aktueller Last > Kapazität fällt aus - bis keine neuen Ausfälle mehr auftreten.
2. **Kritische Toleranz:** kleinstes α, ab dem keine Sekundärausfälle mehr auftreten - je höher, desto verwundbarer das Netz gegenüber dieser Dynamik.
3. **SIR-Ausbreitung (Pastor-Satorras & Vespignani 2001):** jeder infizierte Knoten steckt jeden gesunden Nachbarn unabhängig mit Wahrscheinlichkeit β je Zeitschritt an, genest danach mit
   Wahrscheinlichkeit μ. Über viele Läufe gemittelt ergibt sich die mittlere Endgröße einer Epidemie.
4. **Epidemieschwelle:** β_c = μ·⟨k⟩/⟨k²⟩ (heterogene Mean-Field-Näherung), aus den GEMESSENEN Gradmomenten des konkreten Graphen - unterhalb von β_c stirbt eine Infektion typischerweise schnell
   aus, oberhalb kann sie sich ausbreiten.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    rows_of_4 = [preset_names[i:i + 4] for i in range(0, len(preset_names), 4)]
    for row in rows_of_4:
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.radio("Instanz", options=list(C.KINDS), format_func=lambda v: C.KIND_LABELS[v], key="kind_select",
                     help="Betriebsnetz: gestörtes Straßenraster oder Zufallsgraph. Skalenfreies Netz: Barabási-Albert, bevorzugte Anbindung. Barbell: Lehrbuchbeispiel, von Hand nachrechenbar.")

    side, blocked, nettype = C.DEFAULT_SIDE, C.DEFAULT_BLOCKED, "grid"
    n_ba, m_ba, m0_ba = C.DEFAULT_N_BA, C.DEFAULT_M_BA, C.DEFAULT_M0_BA
    k_barbell = C.DEFAULT_BARBELL_K

    if kind == "city":
        side = st.slider("Seitenlänge des Rasters", *bounds("side_slider"), value=int(ss["side_slider"]), key="side_widget", on_change=store_from_widget, args=("side_slider",),
                          help="Die Instanz hat Seitenlänge² Kreuzungen.")
        nettype = st.radio("Netztyp", options=list(C.NETTYPES), format_func=lambda v: C.NETTYPE_LABELS[v], key="nettype_widget", on_change=store_from_widget, args=("nettype_select",),
                            index=list(C.NETTYPES).index(ss["nettype_select"]),
                            help="Zufallsgraph = Poisson-artige Gradverteilung (näher an der Mean-Field-Annahme der Epidemieschwellen-Formel als das raumstrukturierte Raster, s. README).")
        if nettype == "grid":
            blocked = st.select_slider("Gesperrter Anteil der Straßen", options=list(C.BLOCKED_OPTIONS), value=float(ss["blocked_select"]), format_func=lambda v: f"{v * 100:.0f} %",
                                        key="blocked_widget", on_change=store_from_widget, args=("blocked_select",))
        else:
            blocked = 0.0
    elif kind == "ba":
        n_ba = st.slider("Zahl der Knoten n", *bounds("nba_slider"), value=int(ss["nba_slider"]), key="nba_widget", on_change=store_from_widget, args=("nba_slider",))
        m0_ba = st.slider("Kerngröße m0 (Kreis aus m0 Knoten)", *bounds("m0ba_slider"), value=int(ss["m0ba_slider"]), key="m0ba_widget", on_change=store_from_widget, args=("m0ba_slider",))
        if ss["mba_slider"] > m0_ba:
            ss["mba_slider"] = m0_ba
            push_to_widget("mba_slider")
        if C.M_BA_MIN < m0_ba:
            m_ba = st.slider("Neue Kanten je Knoten m (bevorzugte Anbindung)", C.M_BA_MIN, m0_ba, value=int(ss["mba_slider"]), key="mba_widget", on_change=store_from_widget, args=("mba_slider",),
                              help="Jeder neue Knoten hängt sich mit m Kanten an m bereits vorhandene Knoten an, proportional zu deren Grad gezogen.")
        else:
            m_ba = C.M_BA_MIN                                        # m0 == M_BA_MIN: keine Wahl möglich (min==max würde den Regler zum Absturz bringen)
    else:
        k_barbell = st.slider("Cliquengröße k (Barbell hat 2k Knoten)", *bounds("kbarbell_slider"), value=int(ss["kbarbell_slider"]), key="kbarbell_widget", on_change=store_from_widget,
                               args=("kbarbell_slider",), help="Zwei vollständige Graphen K_k, durch eine einzelne Brücke verbunden.")

    st.markdown("---")
    mode = st.radio("Dynamik", options=list(C.MODES), format_func=lambda v: C.MODE_LABELS[v], key="mode_select",
                     help="Kaskade: lastbasierter Folgeausfall (Motter & Lai 2002). Ausbreitung: stochastische SIR-Epidemie (Pastor-Satorras & Vespignani 2001).")
    initial_strategy = st.radio("Startknoten", options=list(C.INITIAL_STRATEGIES), format_func=lambda v: C.INITIAL_STRATEGY_LABELS[v], key="initial_strategy_select",
                                 help="Höchste Last: der Knoten mit der größten Betweenness (Motter & Lai: löst am ehesten eine Kaskade aus). Zufall: ein per Seed gezogener Knoten.")

    if mode == "kaskade":
        alpha = st.slider("Toleranz α", *bounds("alpha_slider"), value=float(ss["alpha_slider"]), step=C.ALPHA_STEP, key="alpha_widget", on_change=store_from_widget, args=("alpha_slider",),
                           help="Kapazität = (1+α)·Anfangslast. α=0: keinerlei Puffer, jede Lasterhöhung führt sofort zum Ausfall.")
        beta = C.DEFAULT_BETA
        mu = float(ss["mu_slider"])
        n_runs = int(ss["nruns_slider"])
    else:
        alpha = float(ss["alpha_slider"])
        beta = st.slider("Ansteckungswahrscheinlichkeit β", *bounds("beta_slider"), value=float(ss["beta_slider"]), step=C.BETA_STEP, key="beta_widget", on_change=store_from_widget,
                          args=("beta_slider",), help="Wahrscheinlichkeit, dass ein infizierter Knoten einen gesunden Nachbarn je Zeitschritt ansteckt.")
        mu = st.slider("Genesungswahrscheinlichkeit μ", *bounds("mu_slider"), value=float(ss["mu_slider"]), step=C.MU_STEP, key="mu_widget", on_change=store_from_widget, args=("mu_slider",),
                        help="Wahrscheinlichkeit, dass ein infizierter Knoten je Zeitschritt genest.")
        n_runs = st.slider("Wiederholungen (für Mittelwert/Streuband)", *bounds("nruns_slider"), value=int(ss["nruns_slider"]), key="nruns_widget", on_change=store_from_widget,
                            args=("nruns_slider",))

    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)

step = st.select_slider("Schritt", options=list(C.STEPS), key="kas_step", format_func=lambda s: C.STEPS[s])

sync_query_params({"kind_select": kind, "mode_select": mode, "side_slider": int(side) if kind == "city" else int(ss["side_slider"]),
                    "blocked_select": float(blocked) if kind == "city" and nettype == "grid" else float(ss["blocked_select"]), "nettype_select": nettype if kind == "city" else ss["nettype_select"],
                    "nba_slider": int(n_ba) if kind == "ba" else int(ss["nba_slider"]), "m0ba_slider": int(m0_ba) if kind == "ba" else int(ss["m0ba_slider"]),
                    "mba_slider": int(m_ba) if kind == "ba" else int(ss["mba_slider"]), "kbarbell_slider": int(k_barbell) if kind == "barbell" else int(ss["kbarbell_slider"]),
                    "initial_strategy_select": initial_strategy, "alpha_slider": float(alpha), "beta_slider": float(beta), "mu_slider": float(mu), "nruns_slider": int(n_runs),
                    "seed_input": int(seed), "kas_step": int(step)})

base_settings = dict(kind=kind, side=int(side), blocked=float(blocked), nettype=nettype, n_ba=int(n_ba), m_ba=int(m_ba), m0_ba=int(m0_ba), k_barbell=int(k_barbell),
                      initial_strategy=initial_strategy, alpha=float(alpha), beta=float(beta), mu=float(mu), n_runs=int(n_runs), seed=int(seed))
settings = ev.Settings(mode=mode, **base_settings)

if mode == "kaskade":
    with st.spinner("Rechne Kaskade..."):
        inst, ca = _cascade_analysis(settings)
    adj = A.adjacency(inst.n, inst.edges)
    st.markdown("## 🎯 Die Instanz und ihre Kaskade")
    st.markdown(f"**{_german(inst.n)} Knoten, {_german(inst.m)} Kanten**, Startknoten **{ca.initial}** ({C.INITIAL_STRATEGY_LABELS[initial_strategy]}), Toleranz **α={ca.alpha:.2f}**. "
                f"Gesamtausfallanteil G={ca.G:.3f} ({_german(round(ca.G * inst.n))} von {_german(inst.n)} Knoten) nach {len(ca.rounds) - 1} Sekundärrunde(n).")
else:
    with st.spinner("Rechne Ausbreitung..."):
        inst, sa = _sir_analysis(settings)
    adj = A.adjacency(inst.n, inst.edges)
    st.markdown("## 🎯 Die Instanz und ihre Ausbreitung")
    beta_c_txt = f"{sa.beta_c:.4f}" if sa.beta_c is not None else "n/a"
    st.markdown(f"**{_german(inst.n)} Knoten, {_german(inst.m)} Kanten**, Patient Null **{sa.patient_zero}** ({C.INITIAL_STRATEGY_LABELS[initial_strategy]}), β={sa.beta:.2f}, μ={sa.mu:.2f}. "
                f"Mittlere Endgröße über {sa.n_runs} Läufe: {sa.stats['mean']:.1f} von {_german(inst.n)} Knoten. Pastor-Satorras-Schwelle β_c={beta_c_txt}.")

if step == 1:
    if mode == "kaskade":
        n_rounds = len(ca.rounds)
        round_idx = st.slider("Runde", 0, n_rounds - 1, value=0, key=f"round_slider_{kind}_{seed}_{alpha}_{initial_strategy}",
                               help="Runde 0 = der Anfangsausfall selbst, vor jeder Lastumverteilung.") if n_rounds > 1 else 0
        failed = set()
        for r in ca.rounds[:round_idx + 1]:
            failed |= set(r["newly_failed"])
        alive = [v not in failed for v in range(inst.n)]
        giant = A.alive_components(adj, alive)
        giant_nodes = giant[0] if giant else []
        st.plotly_chart(viz.build_cascade_map(inst.xy, [(u, v) for u, v, _ in inst.edges], failed, giant_nodes), width="stretch",
                         key=f"s1_cascade_map_{kind}_{seed}_{alpha}_{initial_strategy}_{round_idx}")
        st.caption(f"Nach Runde {round_idx} von {n_rounds - 1}: {len(failed)} von {inst.n} Knoten ausgefallen, größte Restkomponente {len(giant_nodes)} Knoten.")
    else:
        result = A.sir_simulate(adj, sa.beta, sa.mu, seed, sa.patient_zero, C.MAX_STEPS_SIR, record_states=True)
        n_steps = len(result.states)
        t = st.slider("Zeitschritt", 0, n_steps - 1, value=0, key=f"t_slider_{kind}_{seed}_{beta}_{mu}_{initial_strategy}") if n_steps > 1 else 0
        state = result.states[t]
        st.plotly_chart(viz.build_sir_map(inst.xy, [(u, v) for u, v, _ in inst.edges], state), width="stretch", key=f"s1_sir_map_{kind}_{seed}_{beta}_{mu}_{initial_strategy}_{t}")
        s_c, i_c, r_c = result.trace[t]
        st.caption(f"Zeitschritt {t} von {n_steps - 1}: {s_c} anfällig, {i_c} infiziert, {r_c} genesen ({s_c + i_c + r_c} = n, Erhaltungssatz). Endgröße dieses Laufs: {result.outbreak_size}.")
elif step == 2:
    if mode == "kaskade":
        with st.spinner("Rechne α-Sweep..."):
            _, start, rows, crit = _cascade_alpha_sweep(settings)
        st.plotly_chart(viz.build_cascade_alpha_curve(rows, inst.n, crit), width="stretch", key=f"s2_curve_{kind}_{seed}_{initial_strategy}")
        # Vervielfachungsfaktor direkt aus dem bereits berechneten Sweep abgeleitet (single_removal_damage ist immer 1, s. ev.cascade_vs_single_removal) - KEIN zweiter teurer Sweep-Aufruf.
        mult_rows = [{"alpha": r["alpha"], "multiplier": r["G"] * inst.n} for r in rows]
        # nur ausgewaehlte alpha-Werte fuer die Balken (sonst zu viele Balken)
        display_rows = ev.nearest_alpha_rows(mult_rows, (0.0, 0.2, 0.5, 1.0, 2.0, 5.0))
        st.plotly_chart(viz.build_multiplier_bars(display_rows), width="stretch", key=f"s2_mult_{kind}_{seed}_{initial_strategy}")
        crit_txt = f"α*={crit:.2f}" if crit is not None else "im Sweep nicht erreicht"
        st.caption(f"Kritische Toleranz: {crit_txt}. Bei α={C.DEFAULT_ALPHA:.2f} fallen {_german(round(ca.G * inst.n))} Knoten aus - das {ca.G * inst.n:.1f}-fache des einmaligen Ausfalls (1 "
                   f"Knoten, Stück 8).")
    else:
        st.caption("Diese Ansicht (Kaskadengröße über die Toleranz) gilt nur im Kaskade-Modus - links den Dynamik-Regler auf 'Kaskade' umstellen.")
elif step == 3:
    if mode == "ausbreitung":
        with st.spinner("Rechne β-Sweep..."):
            _, _pz, rows, beta_c = _sir_beta_sweep(settings, BETA_SWEEP_DEFAULT)
        measured = measured_threshold(rows)
        st.plotly_chart(viz.build_sir_beta_curve(rows, inst.n, beta_c, measured), width="stretch", key=f"s3_curve_{kind}_{seed}_{mu}_{n_runs}_{initial_strategy}")
        beta_c_txt = f"{beta_c:.4f}" if beta_c is not None else "n/a"
        measured_txt = f"{measured:.4f}" if measured is not None else "im Sweep nicht erreicht"
        gap_txt = f"{measured - beta_c:+.4f}" if (measured is not None and beta_c is not None) else "n/a"
        st.caption(f"Pastor-Satorras-Vorhersage β_c={beta_c_txt} (aus ⟨k⟩, ⟨k²⟩ des Graphen). Gemessene Schwelle (mittlere Endgröße erreicht {C.THRESHOLD_FRAC:.0%} von n) β*={measured_txt} "
                   f"(Abstand {gap_txt}).")
    else:
        st.caption("Diese Ansicht (Endgröße der Ausbreitung über β) gilt nur im Ausbreitung-Modus - links den Dynamik-Regler auf 'Ausbreitung' umstellen.")
else:
    city_settings = ev.Settings(mode=mode, kind="city", side=int(side) if kind == "city" else C.DEFAULT_SIDE, blocked=float(blocked) if kind == "city" else C.DEFAULT_BLOCKED,
                                 nettype=nettype if kind == "city" else "random", initial_strategy=initial_strategy, mu=float(mu), n_runs=int(n_runs), seed=int(seed))
    ba_settings = ev.Settings(mode=mode, kind="ba", n_ba=int(n_ba) if kind == "ba" else C.DEFAULT_N_BA, m_ba=int(m_ba) if kind == "ba" else C.DEFAULT_M_BA,
                               m0_ba=int(m0_ba) if kind == "ba" else C.DEFAULT_M0_BA, initial_strategy=initial_strategy, mu=float(mu), n_runs=int(n_runs), seed=int(seed))
    barbell_settings = ev.Settings(mode=mode, kind="barbell", k_barbell=int(k_barbell) if kind == "barbell" else C.DEFAULT_BARBELL_K, initial_strategy=initial_strategy, mu=float(mu),
                                    n_runs=int(n_runs), seed=int(seed))
    with st.spinner("Rechne Netzvergleich (Kaskade UND Ausbreitung)..."):
        rows = _network_comparison(city_settings, ba_settings, barbell_settings)
    st.markdown("**Beide Dynamiken je Netz** - die Kernaussage: skalenfrei ist für BEIDE besonders verwundbar (größte kritische Toleranz UND niedrigste Epidemieschwelle).")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.build_network_comparison_bars(rows, "critical_tolerance", "Kritische Toleranz α* (Kaskade, höher = verwundbarer)", "α*"), width="stretch",
                         key=f"s4_crit_{seed}_{side}_{blocked}_{nettype}_{n_ba}_{m_ba}_{m0_ba}_{k_barbell}_{initial_strategy}")
    with c2:
        st.plotly_chart(viz.build_network_comparison_bars(rows, "beta_c", "Pastor-Satorras-Schwelle β_c (Ausbreitung, niedriger = verwundbarer)", "β_c"), width="stretch",
                         key=f"s4_betac_{seed}_{side}_{blocked}_{nettype}_{n_ba}_{m_ba}_{m0_ba}_{k_barbell}_{initial_strategy}_{mu}")
    st.dataframe([{"Netz": r["label"], "n": r["n"], "m": r["m"], "Kritische Toleranz α*": r["critical_tolerance"], "G bei α=0.5": round(r["G_at_default_alpha"], 4),
                   "β_c (Vorhersage)": round(r["beta_c"], 4) if r["beta_c"] is not None else None, "gemessene Schwelle β*": round(r["measured_threshold"], 4) if r["measured_threshold"] is not None
                   else None} for r in rows], width="stretch", hide_index=True)
    most_vulnerable = max((r for r in rows if r["critical_tolerance"] is not None), key=lambda r: r["critical_tolerance"], default=None)
    if most_vulnerable:
        st.caption(f"Größte kritische Toleranz (am kaskaden-verwundbarsten): **{most_vulnerable['label']}**.")

st.markdown("---")

st.markdown("## 🎯 Was die Dynamik verrät")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Knoten", _german(inst.n))
r2.metric("Kanten", _german(inst.m))
if mode == "kaskade":
    r3.metric("Gesamtausfallanteil G", f"{ca.G:.3f}")
    r4.metric("Sekundärrunden", len(ca.rounds) - 1)
else:
    r3.metric("Mittlere Endgröße", f"{sa.stats['mean']:.1f}")
    r4.metric("β_c (Pastor-Satorras)", f"{sa.beta_c:.4f}" if sa.beta_c is not None else "n/a")

st.markdown("---")

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Kapazität_i = (1+α)·Anfangslast_i** | Hat ein Knoten Anfangslast 0 (z. B. ein Blatt), bleibt seine Kapazität für JEDES α exakt 0 - trägt er später auch nur minimal Last, fällt er aus. Auf den drei Vehikeln dieser Demo empirisch nicht beobachtet (s. README), aber ein bekanntes Merkmal der Motter-Lai-Formel. | - |
| **β_c=μ·⟨k⟩/⟨k²⟩ ist eine Mean-Field-Näherung** | Auf räumlich strukturierten, niedrigdimensionalen Netzen (reines Straßenraster) weicht die gemessene Schwelle deutlich stärker von der Formel ab als auf einem Zufallsgraph gleicher Kantenzahl - ein strukturelles Manko der Näherung, kein Rauschen (s. README "Design-Entscheidung"). | - |
| **"Kritische Toleranz" ist hier binär definiert** | Als "keine Sekundärausfälle" - nicht als ein bestimmter Prozentsatz an Gesamtschaden. Ein Netz mit vielen kleinen, aber knapp unterhalb der Kaskaden-Schwelle liegenden Sekundärausfällen zählt hier noch als "kaskadierend". | - |
| **SIR ohne Immunität/Wiederansteckung** | Genesene (R) können sich nie wieder anstecken (SIR, nicht SIS) - für viele reale Infektionskrankheiten mit nachlassender Immunität eine Vereinfachung. | - |
| **Synthetische Instanzen** | Betriebsnetz, skalenfreies Netz und Barbell sind erzeugt, keine echten Infrastruktur- oder Kontaktnetzdaten. | - |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lastbasierte Kaskade (Motter & Lai 2002).** Anfangslast $L_i(0)$ = Betweenness-Zentralität von Knoten $i$ auf dem vollen Graphen. Kapazität $C_i = (1+\alpha)\,L_i(0)$. Nach jedem Ausfall wird
die Betweenness auf dem Restgraphen neu berechnet ($L_i(t)$); jeder überlebende Knoten mit $L_i(t) > C_i$ fällt in Runde $t{+}1$ aus. Terminiert nach höchstens $n$ Runden (die ausgefallene Menge
wächst streng monoton).

**Kritische Toleranz.** $\alpha^* = \min\{\alpha : \text{keine Sekundärausfälle bei Toleranz } \alpha\}$ - je größer $\alpha^*$, desto verwundbarer das Netz gegenüber dieser Dynamik.

**SIR-Ausbreitung (Pastor-Satorras & Vespignani 2001).** Diskrete Zeit, Zustände S (anfällig), I (infiziert), R (genesen). Je Zeitschritt: jeder infizierte Knoten steckt jeden anfälligen Nachbarn
unabhängig mit Wahrscheinlichkeit $\beta$ an, genest danach mit Wahrscheinlichkeit $\mu$ (dieselbe Vorher-Momentaufnahme für beide Ereignisse). Erhaltungssatz: $S(t)+I(t)+R(t)=n$ für alle $t$.

**Epidemieschwelle (heterogene Mean-Field-Näherung).** $\beta_c = \mu\,\dfrac{\langle k \rangle}{\langle k^2 \rangle}$, aus den GEMESSENEN Gradmomenten des konkreten Graphen. Für breite
Gradverteilungen (skalenfreie Netze: $\langle k^2 \rangle$ groß) ist $\beta_c$ viel kleiner als für enge, homogene Gradverteilungen.

**Barbell-Graph.** Zwei vollständige Graphen $K_k$, durch eine Brücke verbunden. Entfernt man ein Brückenende, sind beide Cliquen SOFORT getrennt; innerhalb eines vollständigen Graphen trägt kein
Knoten je Last (jeder kürzeste Weg ist eine direkte Kante) - die Kaskade endet exakt nach Runde 1, unabhängig von $\alpha$.

**Literatur.** Motter, A. E., & Lai, Y.-C. (2002). *Cascade-based attacks on complex networks.* Physical Review E 66, 065102(R). Pastor-Satorras, R., & Vespignani, A. (2001). *Epidemic spreading
in scale-free networks.* Physical Review Letters 86(14), 3200–3203. Brandes, U. (2001). *A faster algorithm for betweenness centrality.* Journal of Mathematical Sociology 25(2), 163–177.

Implementiert in `kas_algorithm.py` (Kaskade, SIR, Epidemieschwelle, Bausteine), `kas_scenario.py` (Instanzen), `kas_evaluation.py` (Analyse, Sweeps, Netzvergleich).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
