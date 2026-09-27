"""Bausteine (`adjacency`, `bfs_distances`, `degree_centrality`, `betweenness_brandes` + Hilfsfunktionen, `components`) sind wortgleiche Kopien aus `rob_algorithm.py` (dort schon aus
`cen_algorithm.py` übernommen). Darauf zwei NEUE Dynamiken:

**Lastbasierte Kaskade** (Motter & Lai 2002, Phys. Rev. E 66, 065102(R)): `motter_lai_cascade` - Anfangslast je Knoten ist seine Betweenness auf dem VOLLEN Graphen (einmal berechnet), Kapazität_i =
(1+α)·Last_i(0). Rundenweise: Betweenness auf dem Restgraphen (nur noch lebende Knoten) neu berechnen, jeder lebende Knoten mit aktueller Last > Kapazität fällt aus, bis keine neuen Ausfälle mehr
auftreten (oder die Sicherheitsgrenze n Runden erreicht ist - die aber wegen der strengen Monotonie, s. Tests, nie tatsächlich gebraucht wird). `critical_tolerance` sucht im gegebenen α-Sweep den
kleinsten Wert, ab dem die Kaskade schon nach der Anfangsrunde stoppt (keine Sekundärausfälle - Motter & Lais binäres "bricht die Lawine aus oder nicht").

**SIR-Ausbreitung** (Pastor-Satorras & Vespignani 2001, Phys. Rev. Lett. 86(14), 3200-3203): `sir_simulate` - diskrete Zeit, jeder infizierte Knoten steckt jeden gesunden Nachbarn unabhängig mit
Wahrscheinlichkeit β je Schritt an, genest danach mit Wahrscheinlichkeit μ; `sir_many_runs` mittelt die Endgröße (= n - S(Ende), also alle je infizierten Knoten - wichtig für den Grenzfall μ=0, wo
niemand je in den R-Zustand wechselt) über viele unabhängige Läufe. `epidemic_threshold_prediction` liefert die heterogene Mean-Field-Schwelle β_c = μ·⟨k⟩/⟨k²⟩ aus den GEMESSENEN Gradmomenten des
konkreten Graphen (nie einer angenommenen Verteilung)."""

import random
from collections import deque
from dataclasses import dataclass, field


# --- Wortgleiche Kopien aus rob_algorithm.py -------------------------------------------------------------------------------------------------------------


def adjacency(n, edges, order="fixed", seed=0):
    """Adjazenzliste aus Kanten (u, v, ...); "fixed" = aufsteigende Nachbarn, "shuffled" = je Knoten gemischt (Seed fest, Python-`random`). Wortgleiche Kopie aus `rob_algorithm.py`."""
    adj = [[] for _ in range(n)]
    for e in edges:
        u, v = int(e[0]), int(e[1])
        adj[u].append(v)
        adj[v].append(u)
    for lst in adj:
        lst.sort()
    if order == "shuffled":
        rng = random.Random(int(seed) * 1_000_003 + 5150)
        for lst in adj:
            rng.shuffle(lst)
    elif order != "fixed":
        raise ValueError(f"unbekannte Reihenfolge {order}")
    return adj


def bfs_distances(adj, s):
    """Abstand von s zu jedem erreichbaren Knoten (-1 = unerreichbar). Gibt (dist, Schritte) zurück. Wortgleiche Kopie aus `rob_algorithm.py`."""
    n = len(adj)
    dist = [-1] * n
    dist[s] = 0
    queue = deque([s])
    steps = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            steps += 1
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
                steps += 1
    return dist, steps


def degree_centrality(adj):
    """Grad je Knoten (Zahl der Nachbarn). Wortgleiche Kopie aus `rob_algorithm.py`."""
    return [len(nbrs) for nbrs in adj]


def _key(u, v):
    return (u, v) if u < v else (v, u)


def _shortest_path_dag(adj, s):
    """BFS-Vorgänger-DAG von s: (dist, sigma, order (Entdeckungsreihenfolge), preds, Schritte). Wortgleiche Kopie aus `rob_algorithm.py`."""
    n = len(adj)
    dist = [-1] * n
    sigma = [0] * n
    dist[s] = 0
    sigma[s] = 1
    preds = [[] for _ in range(n)]
    order = [s]
    queue = deque([s])
    steps = 1
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            steps += 1
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
                order.append(v)
                steps += 1
            if dist[v] == dist[u] + 1:
                sigma[v] += sigma[u]
                preds[v].append(u)
    return dist, sigma, order, preds, steps


@dataclass
class BrandesRun:
    source: int
    dist: list
    sigma: list
    order: list             # BFS-Entdeckungsreihenfolge (Vorwärtsphase)
    preds: list
    delta: list = field(default_factory=list)          # Abhängigkeit je Knoten NACH der Rückwärtsphase
    edge_delta: dict = field(default_factory=dict)      # Kanten-Abhängigkeit dieser einen Quelle
    steps: int = 0


def brandes_source(adj, s):
    """Ein Brandes-Durchlauf von einer einzigen Quelle s. Wortgleiche Kopie aus `rob_algorithm.py`."""
    dist, sigma, order, preds, steps = _shortest_path_dag(adj, s)
    n = len(adj)
    delta = [0.0] * n
    edge_delta = {}
    for w in reversed(order):
        coeff = (1.0 + delta[w]) / sigma[w]
        steps += 1
        for v in preds[w]:
            steps += 1
            c = sigma[v] * coeff
            delta[v] += c
            e = _key(v, w)
            edge_delta[e] = edge_delta.get(e, 0.0) + c
    return BrandesRun(s, dist, sigma, order, preds, delta, edge_delta, steps)


def betweenness_brandes(adj):
    """Brandes (2001): eine BFS je Startknoten (O(n*m) insgesamt), liefert Knoten- UND Kanten-Betweenness (unnormiert). Gibt (Knoten-Liste, Kanten-Dict, Elementarschritte) zurück. Wortgleiche Kopie
    aus `rob_algorithm.py`. Ein Knoten ohne Nachbarn (z. B. ein in dieser Runde bereits ausgefallener Knoten, dessen Zeile in der Adjazenzliste leer ist) durchläuft eine triviale Ein-Knoten-BFS und
    trägt rechnerisch 0 bei - so liefert dieselbe Funktion unverändert auch die Restgraph-Betweenness, ohne die Knoten neu durchzunummerieren."""
    n = len(adj)
    node_raw = [0.0] * n
    edge_raw = {}
    steps = 0
    for s in range(n):
        run = brandes_source(adj, s)
        steps += run.steps
        for v in range(n):
            if v != s:
                node_raw[v] += run.delta[v]
        for e, c in run.edge_delta.items():
            edge_raw[e] = edge_raw.get(e, 0.0) + c
    node_between = [v * 0.5 for v in node_raw]
    edge_between = {e: c * 0.5 for e, c in edge_raw.items()}
    return node_between, edge_between, steps


def components(adj):
    """Liste der Komponentengrößen (Reihenfolge = Fundreihenfolge des kleinsten Knotens jeder Komponente), über wiederholte BFS. Wortgleiche Kopie aus `rob_algorithm.py`."""
    n = len(adj)
    seen = [False] * n
    sizes = []
    for s in range(n):
        if seen[s]:
            continue
        dist, _ = bfs_distances(adj, s)
        comp = [v for v, d in enumerate(dist) if d >= 0]
        for v in comp:
            seen[v] = True
        sizes.append(len(comp))
    return sizes


# --- Lastbasierte Kaskade (Motter & Lai 2002) ------------------------------------------------------------------------------------------------------------


def _sub_adjacency(adj, alive):
    """Adjazenzliste des Restgraphen: tote Knoten (alive[v]=False) bekommen eine LEERE Nachbarliste (statt entfernt/umnummeriert zu werden) - so bleiben Knotenindizes über alle Runden stabil und
    `betweenness_brandes` liefert direkt die Restgraph-Betweenness (isolierte Knoten tragen 0 bei, s. dort)."""
    n = len(adj)
    return [sorted(u for u in adj[v] if alive[u]) if alive[v] else [] for v in range(n)]


def alive_components(adj, alive):
    """Liste der Komponenten (je eine sortierte Liste lebender Knoten) NUR unter den lebenden Knoten, größte zuerst, per BFS auf dem Restgraphen - tote Knoten (alive[v]=False) zählen nicht als
    eigene Ein-Knoten-Komponente mit (anders als das rohe `components()`). Für die "in Aktion"-Karte (Schritt 1 der App): welche Knoten gehören zur größten Restkomponente."""
    sub = _sub_adjacency(adj, alive)
    n = len(sub)
    seen = [False] * n
    comps = []
    for s in range(n):
        if not alive[s] or seen[s]:
            continue
        dist, _ = bfs_distances(sub, s)
        comp = sorted(v for v, d in enumerate(dist) if d >= 0 and alive[v])
        for v in comp:
            seen[v] = True
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def _alive_giant_size(sub_adj, alive):
    """Größte Komponente NUR unter den lebenden Knoten (tote Knoten - leere Nachbarliste - zählen nicht als eigene Ein-Knoten-Komponente mit, anders als das rohe `components()`)."""
    n = len(sub_adj)
    seen = [False] * n
    best = 0
    for s in range(n):
        if not alive[s] or seen[s]:
            continue
        dist, _ = bfs_distances(sub_adj, s)
        comp = [v for v, d in enumerate(dist) if d >= 0 and alive[v]]
        for v in comp:
            seen[v] = True
        best = max(best, len(comp))
    return best


def motter_lai_cascade(adj, alpha, initial_removed):
    """Motter & Lai (2002): Anfangslast Last_i(0) = Betweenness auf dem VOLLEN Graphen (einmal berechnet); Kapazität_i = (1+α)·Last_i(0). Rundenweise: Betweenness auf dem Restgraphen (nur noch
    lebende Knoten) neu berechnen; jeder lebende Knoten mit aktueller Last > Kapazität fällt aus. Wiederholt, bis keine neuen Ausfälle mehr auftreten oder die Sicherheitsgrenze n Runden erreicht
    ist (die wegen der strengen Monotonie - jede fortgesetzte Runde entfernt mindestens 1 weiteren Knoten von höchstens n - nie tatsächlich erreicht wird, s. Tests).

    Gibt (rounds, G) zurück: `rounds` ist eine Liste von {"round": r, "newly_failed": [...], "giant_size": int} (Runde 0 = der Anfangsausfall selbst, vor jeder Lastumverteilung); `G` ist der
    Gesamtausfallanteil (Anfangsausfall + alle Sekundärausfälle) / n."""
    n = len(adj)
    alpha = float(alpha)
    initial_removed = sorted(set(int(v) for v in initial_removed))
    if any(not (0 <= v < n) for v in initial_removed):
        raise ValueError("initial_removed außerhalb des gültigen Knotenbereichs")
    node_between0, _, _ = betweenness_brandes(adj)
    capacity = [(1.0 + alpha) * node_between0[v] for v in range(n)]
    alive = [True] * n
    for v in initial_removed:
        alive[v] = False

    rounds = [{"round": 0, "newly_failed": list(initial_removed), "giant_size": _alive_giant_size(_sub_adjacency(adj, alive), alive)}]
    total_failed = set(initial_removed)
    safety_bound = n
    round_idx = 0
    tol = 1e-9                                          # Toleranz gegen Gleitkommarauschen genau an der Kapazitätsgrenze (z. B. Last==Kapazität exakt)
    while round_idx < safety_bound:
        round_idx += 1
        cur = _sub_adjacency(adj, alive)
        node_between_cur, _, _ = betweenness_brandes(cur)
        newly_failed = [v for v in range(n) if alive[v] and node_between_cur[v] > capacity[v] + tol]
        if not newly_failed:
            break
        for v in newly_failed:
            alive[v] = False
        total_failed.update(newly_failed)
        rounds.append({"round": round_idx, "newly_failed": newly_failed, "giant_size": _alive_giant_size(_sub_adjacency(adj, alive), alive)})
    G = len(total_failed) / n if n else 0.0
    return rounds, G


def critical_tolerance(adj, initial_removed, alphas):
    """Kleinstes α aus dem gegebenen (aufsteigend sortierten) Sweep, ab dem KEINE globale Kaskade mehr auftritt - hier operationalisiert als "die Kaskade endet schon nach der Anfangsrunde, keine
    Sekundärausfälle" (Motter & Lais binäres Bild: die Lawine bricht aus oder nicht). Gibt `None` zurück, wenn selbst das größte α im Sweep noch eine Kaskade auslöst."""
    for alpha in sorted(float(a) for a in alphas):
        rounds, _ = motter_lai_cascade(adj, alpha, initial_removed)
        if len(rounds) == 1:
            return alpha
    return None


# --- SIR-Ausbreitung (Pastor-Satorras & Vespignani 2001) ---------------------------------------------------------------------------------------------


@dataclass
class SIRResult:
    trace: list                  # Liste von (s_count, i_count, r_count), ein Eintrag je Zeitschritt inkl. t=0
    final_s: int
    final_i: int
    final_r: int
    outbreak_size: int           # n - final_s: JE infiziert gewesene Knoten (I + R am Ende) - robust auch fuer mu=0 (dort bleibt final_r=0 fuer immer)
    steps_run: int
    states: list = field(default_factory=list)   # nur gefuellt, wenn record_states=True: eine Kopie des Zustandsvektors je Zeitschritt (fuer die "in Aktion"-Karte, Schritt 1 der App)


def sir_simulate(adj, beta, mu, seed, patient_zero, max_steps=200, record_states=False):
    """Diskrete Zeit, S/I/R-Zustände. Je Zeitschritt (synchrones Update auf Basis des Zustands VOR diesem Schritt): jeder infizierte Knoten steckt jeden gesunden (S) Nachbarn unabhängig mit
    Wahrscheinlichkeit β an; anschließend genest jeder infizierte Knoten (derselbe Vorher-Zustand) mit Wahrscheinlichkeit μ. `random.Random`-Hauskonvention (nicht numpy) für plattformstabile
    Reproduzierbarkeit. Erhaltungssatz per Konstruktion: S+I+R == n in JEDEM Zeitschritt (es werden nur Zustände umbenannt, nie Knoten hinzugefügt/entfernt). `record_states=True` speichert
    zusätzlich den vollen Zustandsvektor je Schritt (nur für die Einzel-Lauf-Karte in der App gebraucht, `sir_many_runs` braucht das nicht und lässt es aus)."""
    n = len(adj)
    beta = float(beta)
    mu = float(mu)
    patient_zero = int(patient_zero)
    if not (0 <= patient_zero < n):
        raise ValueError("patient_zero außerhalb des gültigen Knotenbereichs")
    rng = random.Random(int(seed) * 1_000_003 + 6421)
    state = [0] * n                          # 0=S, 1=I, 2=R
    state[patient_zero] = 1
    trace = [(state.count(0), state.count(1), state.count(2))]
    states = [list(state)] if record_states else []
    steps_run = 0
    for _ in range(int(max_steps)):
        infected = [v for v in range(n) if state[v] == 1]
        if not infected:
            break
        newly_infected = set()
        for v in infected:
            for u in adj[v]:
                if state[u] == 0 and u not in newly_infected and rng.random() < beta:
                    newly_infected.add(u)
        newly_recovered = [v for v in infected if rng.random() < mu]
        for u in newly_infected:
            state[u] = 1
        for v in newly_recovered:
            state[v] = 2
        steps_run += 1
        trace.append((state.count(0), state.count(1), state.count(2)))
        assert trace[-1][0] + trace[-1][1] + trace[-1][2] == n
        if record_states:
            states.append(list(state))
    final_s, final_i, final_r = trace[-1]
    return SIRResult(trace, final_s, final_i, final_r, n - final_s, steps_run, states)


def sir_many_runs(adj, beta, mu, n_runs, seed, patient_zero, max_steps=200):
    """Endgröße (n - S(Ende), s. `SIRResult.outbreak_size`) über `n_runs` unabhängige stochastische Läufe (jeder Lauf ein eigener, aus `seed` und dem Laufindex abgeleiteter Startzustand des
    `random.Random`-Stroms). Gibt Mittelwert, Median, 10./90.-Perzentil und die rohe Liste zurück."""
    n_runs = int(n_runs)
    sizes = []
    for i in range(n_runs):
        run_seed = int(seed) * 1013 + i
        result = sir_simulate(adj, beta, mu, run_seed, patient_zero, max_steps)
        sizes.append(result.outbreak_size)
    sizes.sort()

    def percentile(p):
        if not sizes:
            return 0
        idx = min(len(sizes) - 1, max(0, int(round(p / 100 * (len(sizes) - 1)))))
        return sizes[idx]

    mean = sum(sizes) / len(sizes) if sizes else 0.0
    return {"mean": mean, "p10": percentile(10), "p50": percentile(50), "p90": percentile(90), "runs": sizes}


def epidemic_threshold_prediction(adj, mu):
    """β_c = μ·⟨k⟩/⟨k²⟩ (heterogene Mean-Field-Näherung, Pastor-Satorras & Vespignani 2001) - aus den GEMESSENEN Gradmomenten ⟨k⟩, ⟨k²⟩ des konkreten Graphen (nie einer angenommenen Verteilung).
    Gibt `None` zurück, wenn ⟨k²⟩=0 (kantenloser Graph, keine Ausbreitung möglich, die Formel hätte einen Nenner von 0)."""
    n = len(adj)
    if n == 0:
        return None
    degrees = [len(a) for a in adj]
    mean_k = sum(degrees) / n
    mean_k2 = sum(d * d for d in degrees) / n
    if mean_k2 <= 0:
        return None
    return float(mu) * mean_k / mean_k2
