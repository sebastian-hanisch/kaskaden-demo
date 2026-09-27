"""Auswertung: Analyse einer Instanz im gewählten Modus (Kaskade ODER Ausbreitung), α-Sweep samt kritischer Toleranz, Kaskade gegen einmaligen Ausfall (Vervielfachungsfaktor gegenüber Stück 8),
β-Sweep der Endgröße samt gemessener Epidemieschwelle gegen die Pastor-Satorras-Formel, Drei-Netze-Vergleich (Kernaussage: skalenfrei ist für BEIDE Dynamiken besonders verwundbar)."""

from dataclasses import dataclass

import kas_algorithm as A
import kas_constants as C
import kas_scenario as S


@dataclass
class Settings:
    kind: str = "city"                   # "city" | "ba" | "barbell"
    mode: str = "kaskade"                # "kaskade" | "ausbreitung"
    # Betriebsnetz
    side: int = C.DEFAULT_SIDE
    blocked: float = C.DEFAULT_BLOCKED
    nettype: str = "grid"                # "grid" | "random"
    # Skalenfreies Netz
    n_ba: int = C.DEFAULT_N_BA
    m_ba: int = C.DEFAULT_M_BA
    m0_ba: int = C.DEFAULT_M0_BA
    # Barbell-Lehrbuch
    k_barbell: int = C.DEFAULT_BARBELL_K
    # Kaskade
    alpha: float = C.DEFAULT_ALPHA
    initial_strategy: str = "highest_load"     # "highest_load" | "random"
    # Ausbreitung
    beta: float = C.DEFAULT_BETA
    mu: float = C.DEFAULT_MU
    n_runs: int = C.DEFAULT_N_RUNS
    max_steps: int = C.MAX_STEPS_SIR
    # gemeinsam
    seed: int = C.DEFAULT_SEED


def instance(settings):
    if settings.kind == "city":
        return S.generate(settings.side, settings.blocked, settings.nettype, settings.seed)
    if settings.kind == "ba":
        return S.barabasi_albert_instance(settings.n_ba, settings.m_ba, settings.m0_ba, settings.seed)
    if settings.kind == "barbell":
        return S.barbell_instance(settings.k_barbell)
    raise ValueError(f"unbekannte Instanzart {settings.kind}")


def initial_node(adj, strategy, seed):
    """Der Startknoten des Ausfalls/der Infektion: `highest_load` = höchste Betweenness auf dem vollen Graphen (Motter & Lai: genau der Knoten, dessen Ausfall am ehesten eine Kaskade auslöst),
    `random` = ein per Seed gezogener Zufallsknoten."""
    n = len(adj)
    if strategy == "highest_load":
        node_between, _, _ = A.betweenness_brandes(adj)
        return max(range(n), key=lambda v: (node_between[v], -v))
    if strategy == "random":
        import random
        rng = random.Random(int(seed) * 1_000_003 + 2029)
        return rng.randrange(n)
    raise ValueError(f"unbekannte Anfangsstrategie {strategy}")


# --- Kaskade (Motter & Lai 2002) -------------------------------------------------------------------------------------------------------------------------


@dataclass
class CascadeAnalysis:
    n: int
    m: int
    initial: int
    alpha: float
    rounds: list
    G: float                              # Gesamtausfallanteil (Anfangsausfall + Sekundärausfälle)


def analyse_cascade(settings):
    inst = instance(settings)
    adj = A.adjacency(inst.n, inst.edges)
    start = initial_node(adj, settings.initial_strategy, settings.seed)
    rounds, G = A.motter_lai_cascade(adj, settings.alpha, [start])
    return inst, CascadeAnalysis(inst.n, inst.m, start, float(settings.alpha), rounds, G)


def cascade_alpha_sweep(settings, alphas=C.ALPHA_SWEEP):
    """G(α)-Kurve auf derselben Instanz/demselben Startknoten, plus die kritische Toleranz (kleinstes α ohne Sekundärausfälle - direkt aus den ohnehin berechneten Sweep-Zeilen abgelesen, KEINE
    zweite unabhängige Kaskaden-Serie: das spart bei teuren Instanzen (n groß) die Hälfte der Rechenzeit gegenüber einem separaten `A.critical_tolerance`-Aufruf über denselben Sweep)."""
    inst = instance(settings)
    adj = A.adjacency(inst.n, inst.edges)
    start = initial_node(adj, settings.initial_strategy, settings.seed)
    rows = []
    for alpha in alphas:
        rounds, G = A.motter_lai_cascade(adj, alpha, [start])
        rows.append({"alpha": alpha, "G": G, "n_rounds": len(rounds), "giant_final": rounds[-1]["giant_size"]})
    crit = next((r["alpha"] for r in sorted(rows, key=lambda r: r["alpha"]) if r["n_rounds"] == 1), None)
    return inst, start, rows, crit


def cascade_vs_single_removal(settings, alphas=C.ALPHA_SWEEP):
    """Vervielfachungsfaktor: Schaden der Kaskadendynamik (G·n = Gesamtausfall) gegen den Schaden EINES einmaligen, isolierten Ausfalls desselben Startknotens (Stück 8: dort verschwindet nur er
    selbst, keine Folgeausfälle - das Vergleichsmaß hier ist "wie viele Knoten fallen zusätzlich aus", 1 für den einmaligen Fall gegen `G·n` für die Kaskade). Baut auf `cascade_alpha_sweep` auf
    (KEINE zweite unabhängige Kaskaden-Serie über denselben Sweep)."""
    inst, start, sweep_rows, _crit = cascade_alpha_sweep(settings, alphas)
    single_removal_damage = 1                                # Stück 8: ein einmaliger Ausfall entfernt GENAU diesen einen Knoten, keine Folgeausfaelle
    rows = []
    for r in sweep_rows:
        cascade_damage = round(r["G"] * inst.n)
        rows.append({"alpha": r["alpha"], "cascade_damage": cascade_damage, "single_removal_damage": single_removal_damage,
                     "multiplier": cascade_damage / single_removal_damage if single_removal_damage else float("nan")})
    return inst, start, rows


def nearest_alpha_rows(rows, targets):
    """Für jeden Zielwert in `targets` die Zeile aus `rows` mit dem NÄCHSTGELEGENEN tatsächlich vorhandenen α (nie exaktes `==`, da ein Sweep mit fester Schrittweite - z. B. ALPHA_SWEEP in
    0.2-Schritten - einen "griffigen" Zielwert wie 0.5 sonst stillschweigend nie trifft und der entsprechende Balken/Punkt in der Anzeige fehlen würde). Gibt die passenden `rows`-Einträge sortiert
    nach α zurück, ohne Duplikate (mehrere Zielwerte können denselben nächstgelegenen Punkt treffen)."""
    if not rows:
        return []
    chosen_alphas = sorted({min(rows, key=lambda r: abs(r["alpha"] - target))["alpha"] for target in targets})
    return [r for r in rows if r["alpha"] in chosen_alphas]


# --- Ausbreitung (Pastor-Satorras & Vespignani 2001) -----------------------------------------------------------------------------------------------------


@dataclass
class SIRAnalysis:
    n: int
    m: int
    patient_zero: int
    beta: float
    mu: float
    n_runs: int
    result_single: object                 # ein einzelner Lauf (fuer die "in Aktion"-Karte, Schritt 1)
    stats: dict                            # sir_many_runs-Ergebnis (mean/p10/p50/p90/runs)
    beta_c: float                          # epidemic_threshold_prediction


def analyse_sir(settings):
    inst = instance(settings)
    adj = A.adjacency(inst.n, inst.edges)
    patient_zero = initial_node(adj, settings.initial_strategy, settings.seed)
    result_single = A.sir_simulate(adj, settings.beta, settings.mu, settings.seed, patient_zero, settings.max_steps)
    stats = A.sir_many_runs(adj, settings.beta, settings.mu, settings.n_runs, settings.seed, patient_zero, settings.max_steps)
    beta_c = A.epidemic_threshold_prediction(adj, settings.mu)
    return inst, SIRAnalysis(inst.n, inst.m, patient_zero, float(settings.beta), float(settings.mu), int(settings.n_runs), result_single, stats, beta_c)


def sir_beta_sweep(settings, betas):
    inst = instance(settings)
    adj = A.adjacency(inst.n, inst.edges)
    patient_zero = initial_node(adj, settings.initial_strategy, settings.seed)
    rows = []
    for beta in betas:
        stats = A.sir_many_runs(adj, beta, settings.mu, settings.n_runs, settings.seed, patient_zero, settings.max_steps)
        rows.append({"beta": beta, "mean": stats["mean"], "p10": stats["p10"], "p50": stats["p50"], "p90": stats["p90"], "mean_fraction": stats["mean"] / inst.n if inst.n else 0.0})
    beta_c = A.epidemic_threshold_prediction(adj, settings.mu)
    return inst, patient_zero, rows, beta_c


def measured_threshold(rows, frac=None):
    """Kleinstes β (linear zwischen den beiden umschließenden Sweep-Punkten interpoliert), bei dem die mittlere Endgrößen-Fraktion erstmals >= `frac` ist - eine endliche-Größe-Proxy-Schwelle fürs
    Kreuzen mit der Pastor-Satorras-Formel (analog zur PERCOLATION_FRAC_VANISH-Design-Entscheidung der Robustheits-Demo: eine kleine, aber nicht zu kleine Schwelle liegt näher an der
    asymptotischen Mean-Field-Vorhersage als ein großer, griffiger Wert). Gibt `None` zurück, wenn `frac` im gegebenen Sweep nie erreicht wird."""
    frac = C.THRESHOLD_FRAC if frac is None else frac
    rows_sorted = sorted(rows, key=lambda r: r["beta"])
    prev = None
    for row in rows_sorted:
        if row["mean_fraction"] >= frac:
            if prev is None:
                return row["beta"]
            b0, f0 = prev["beta"], prev["mean_fraction"]
            b1, f1 = row["beta"], row["mean_fraction"]
            if f1 == f0:
                return b1
            t = (frac - f0) / (f1 - f0)
            return b0 + t * (b1 - b0)
        prev = row
    return None


def threshold_check(settings, betas, frac=None):
    """Gemessene Epidemieschwelle (via `measured_threshold`) gegen β_c = μ·⟨k⟩/⟨k²⟩ (Pastor-Satorras & Vespignani 2001)."""
    frac = C.THRESHOLD_FRAC if frac is None else frac
    inst, patient_zero, rows, beta_c = sir_beta_sweep(settings, betas)
    measured = measured_threshold(rows, frac)
    return {"inst": inst, "patient_zero": patient_zero, "rows": rows, "beta_c": beta_c, "measured": measured, "frac": frac,
            "gap": None if (measured is None or beta_c is None) else measured - beta_c}


# --- Drei Netze im Vergleich (Schritt 4, die Kernaussage) --------------------------------------------------------------------------------------------------


def network_comparison(city_settings, ba_settings, barbell_settings, alphas=C.ALPHA_SWEEP, betas=None):
    """Betriebsnetz/Skalenfreies Netz/Barbell je Kaskade (kritische Toleranz, Gesamtausfall bei einem festen, für alle drei GLEICHEN Referenz-α) und Ausbreitung (gemessene Schwelle gegen β_c,
    unter dem jeweils EIGENEN μ/n_runs der drei Settings-Objekte) - die Kernaussage: skalenfrei ist für BEIDE Dynamiken besonders verwundbar (größte kritische Toleranz UND niedrigste
    Epidemieschwelle)."""
    betas = betas if betas is not None else tuple(round(0.01 * i, 3) for i in range(1, 61))
    rows = []
    for label, settings in (("Betriebsnetz", city_settings), ("Skalenfrei", ba_settings), ("Barbell", barbell_settings)):
        inst = instance(settings)
        adj = A.adjacency(inst.n, inst.edges)
        start = initial_node(adj, settings.initial_strategy, settings.seed)
        crit = A.critical_tolerance(adj, [start], alphas)
        _, G_ref = A.motter_lai_cascade(adj, C.DEFAULT_ALPHA, [start])
        check = threshold_check(settings, betas)
        rows.append({"label": label, "n": inst.n, "m": inst.m, "critical_tolerance": crit, "G_at_default_alpha": G_ref, "beta_c": check["beta_c"], "measured_threshold": check["measured"]})
    return rows
