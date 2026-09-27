"""Zentrale Korrektheits-Kette, Punkte 1-4 (Motter & Lai 2002, lastbasierte Kaskade):

1) Kaskaden-Lastneuberechnung je Runde gegen unabhängige Brandes-Neuberechnung auf dem Restgraphen (kleine Instanzen) - hier zusätzlich gegen networkx als zweite, unabhängige Gegenprobe.
2) Ausgefallene Menge wächst monoton, Kaskade terminiert in <= n Runden (Satz, Test).
3) alpha -> unendlich: Kaskade bleibt exakt beim Anfangsausfall stehen, keine Sekundärausfälle (Grenzfall-Test).
4) Barbell: Kaskade endet nach Runde 1 exakt (Brücke schon getrennt, keine Lastumverteilung mehr möglich) - festgeschrieben."""

import networkx as nx
import pytest

import kas_algorithm as A
import kas_scenario as S


def _nx_graph_from_adj(adj):
    g = nx.Graph()
    g.add_nodes_from(range(len(adj)))
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if u < v:
                g.add_edge(u, v)
    return g


def _small_instances():
    out = []
    for side in (3, 4, 5):
        for seed in (1, 2, 3):
            out.append(S.generate(side, 0.2, "grid", seed))
    for n, m, m0, seed in [(15, 2, 4, 1), (20, 1, 4, 2), (25, 2, 5, 3)]:
        out.append(S.barabasi_albert_instance(n, m, m0, seed))
    for k in (4, 5, 6):
        out.append(S.barbell_instance(k))
    return out


SMALL_INSTANCES = _small_instances()


# --- 1. Lastneuberechnung je Runde gegen unabhängige Brandes-/networkx-Neuberechnung -------------------------------------------------------------------


@pytest.mark.parametrize("inst", SMALL_INSTANCES)
def test_round_by_round_load_matches_independent_recomputation(inst):
    """Rekonstruiert aus dem Rundenprotokoll den Restgraphen NACH jeder Runde (unabhängig vom internen `_sub_adjacency`, direkt aus den Original-Kanten neu aufgebaut) und vergleicht die
    Betweenness dort sowohl mit einer frischen `betweenness_brandes`-Neuberechnung als auch mit networkx - falls die Runde noch weiterläuft (newly_failed der NÄCHSTEN Runde nicht leer), muss die
    hier gemessene Last exakt die sein, die zu den dort ausgefallenen Knoten geführt hat."""
    adj = A.adjacency(inst.n, inst.edges)
    node_between0, _, _ = A.betweenness_brandes(adj)
    initial = [max(range(inst.n), key=lambda v: node_between0[v])]
    alpha = 0.05                                             # kleine Toleranz: provoziert typischerweise mehrere Kaskadenrunden
    rounds, _ = A.motter_lai_cascade(adj, alpha, initial)
    capacity = [(1.0 + alpha) * node_between0[v] for v in range(inst.n)]

    alive = [True] * inst.n
    for v in rounds[0]["newly_failed"]:
        alive[v] = False
    for r in rounds[1:]:
        # unabhaengiger Wiederaufbau des Restgraphen aus den ROHEN Kanten (nicht ueber _sub_adjacency)
        edges = [(u, v) for u, v, _ in inst.edges if alive[u] and alive[v]]
        alive_nodes = [v for v in range(inst.n) if alive[v]]
        remap = {old: i for i, old in enumerate(alive_nodes)}
        sub_adj = [[] for _ in alive_nodes]
        for u, v in edges:
            sub_adj[remap[u]].append(remap[v])
            sub_adj[remap[v]].append(remap[u])
        for lst in sub_adj:
            lst.sort()
        node_between_sub, _, _ = A.betweenness_brandes(sub_adj)
        node_between_full = [0.0] * inst.n
        for old, i in remap.items():
            node_between_full[old] = node_between_sub[i]

        g = _nx_graph_from_adj(sub_adj)
        want_nx = nx.betweenness_centrality(g, normalized=False)

        expected_failed = sorted(v for v in alive_nodes if node_between_full[v] > capacity[v] + 1e-9)
        assert expected_failed == sorted(r["newly_failed"])
        for i, old in enumerate(alive_nodes):
            assert node_between_full[old] == pytest.approx(want_nx[i], abs=1e-6)
        for v in r["newly_failed"]:
            alive[v] = False


# --- 2. Ausgefallene Menge waechst monoton, Terminierung in <= n Runden ---------------------------------------------------------------------------------


@pytest.mark.parametrize("inst", SMALL_INSTANCES)
@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.3, 0.5, 1.0, 2.0])
def test_failed_set_grows_monotonically_and_terminates_within_n_rounds(inst, alpha):
    adj = A.adjacency(inst.n, inst.edges)
    node_between0, _, _ = A.betweenness_brandes(adj)
    initial = [max(range(inst.n), key=lambda v: node_between0[v])]
    rounds, G = A.motter_lai_cascade(adj, alpha, initial)
    assert len(rounds) <= inst.n + 1                      # Runde 0 (Anfangsausfall) + hoechstens n weitere Runden
    seen = set()
    for r in rounds:
        newly = set(r["newly_failed"])
        assert not (newly & seen)                          # jede Runde faellt NUR neue Knoten aus, nie ein zweites Mal
        assert r["round"] == 0 or len(newly) >= 1           # jede fortgesetzte Runde entfernt mindestens 1 weiteren Knoten
        seen |= newly
    assert G == pytest.approx(len(seen) / inst.n)
    # Riesenkomponenten-Groesse ist nicht wachsend ueber die Runden (kann nur schrumpfen, wenn weitere Knoten ausfallen)
    sizes = [r["giant_size"] for r in rounds]
    assert all(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1))


# --- 3. alpha -> unendlich: keine Sekundaerausfaelle (Grenzfall) ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("inst", SMALL_INSTANCES)
def test_huge_alpha_stops_exactly_at_initial_removal(inst):
    """Empirisch verifiziert (s. Scoping-Notiz): bei sehr großem alpha bleibt es auf allen drei Vehikeln (Betriebsnetz, Skalenfrei, Barbell) beim Anfangsausfall - dieselbe Aussage gilt auch fuer
    zufaellig gewaehlte Startknoten (Stichprobenpruefung in test_cascade_random_removal_huge_alpha_no_secondary_failures unten)."""
    adj = A.adjacency(inst.n, inst.edges)
    node_between0, _, _ = A.betweenness_brandes(adj)
    initial = [max(range(inst.n), key=lambda v: node_between0[v])]
    rounds, G = A.motter_lai_cascade(adj, 1e9, initial)
    assert len(rounds) == 1
    assert rounds[0]["newly_failed"] == initial
    assert G == pytest.approx(1.0 / inst.n)


def test_cascade_random_removal_huge_alpha_no_secondary_failures():
    """Stichprobe ueber viele zufaellige (Instanz, Startknoten)-Kombinationen: bei alpha=1e9 endet die Kaskade praktisch immer schon nach der Anfangsrunde. Absichtlich >=95% statt 100%, um die -
    literaturbekannte, in der Modul-Docstring erlaeuterte - theoretische Moeglichkeit eines Knotens mit Anfangslast exakt 0 abzudecken, der nach der Entfernung ploetzlich Last traegt (Kapazitaet
    bleibt dann 0 fuer JEDES alpha); auf den hier verwendeten Instanzen/Groessen tritt das nicht auf (0 Ausreisser gemessen), s. Bericht."""
    ok = 0
    total = 0
    for seed in range(1, 40):
        inst = S.barabasi_albert_instance(20 + (seed % 40), 2, 4, seed)
        adj = A.adjacency(inst.n, inst.edges)
        v = (seed * 7) % inst.n
        rounds, _ = A.motter_lai_cascade(adj, 1e9, [v])
        total += 1
        if len(rounds) == 1:
            ok += 1
    assert total >= 30
    assert ok / total >= 0.95


# --- 4. Barbell: Kaskade endet exakt nach Runde 1 --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 4, 5, 6, 8, 10, 12, 20])
@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.5, 1.0, 3.0])
def test_barbell_cascade_ends_after_round_one_regardless_of_alpha(k, alpha):
    """Wird ein Brueckenende entfernt, ist der Graph SOFORT in zwei vollstaendige Graphen getrennt - dort traegt (per Satz) niemand jemals Last (jeder kuerzeste Weg innerhalb eines vollstaendigen
    Graphen ist eine direkte Kante), die Last aller ueberlebenden Knoten bleibt exakt 0 fuer immer, keine Lastumverteilung ueber die (nicht mehr existierende) Bruecke moeglich - die Kaskade endet
    nach GENAU Runde 1, UNABHAENGIG von alpha (auch bei alpha=0, der strengsten Toleranz)."""
    inst = S.barbell_instance(k)
    adj = A.adjacency(inst.n, inst.edges)
    bridge_left = k - 1
    rounds, G = A.motter_lai_cascade(adj, alpha, [bridge_left])
    assert len(rounds) == 1
    assert rounds[0]["newly_failed"] == [bridge_left]
    assert rounds[0]["giant_size"] == k                     # die unberuehrte rechte Clique bleibt die groesste Komponente
    assert G == pytest.approx(1.0 / inst.n)


def test_barbell_critical_tolerance_is_zero():
    """Direkte Folge von Punkt 4: schon bei alpha=0 (der striktesten Toleranz im Sweep) gibt es keine Sekundaerausfaelle - die kritische Toleranz ist also 0.0."""
    for k in (4, 6, 8, 10):
        inst = S.barbell_instance(k)
        adj = A.adjacency(inst.n, inst.edges)
        alphas = [0.0, 0.1, 0.5, 1.0, 2.0]
        crit = A.critical_tolerance(adj, [k - 1], alphas)
        assert crit == 0.0


def test_critical_tolerance_returns_none_when_sweep_too_small():
    """Auf einer Instanz, die selbst bei grossem alpha noch kaskadiert (kleiner Sweep, der die tatsaechliche kritische Toleranz nicht erreicht), muss `critical_tolerance` ehrlich `None` melden statt
    einen falschen Wert zu erfinden."""
    inst = S.barabasi_albert_instance(40, 3, 5, 7)
    adj = A.adjacency(inst.n, inst.edges)
    node_between0, _, _ = A.betweenness_brandes(adj)
    hub = max(range(inst.n), key=lambda v: node_between0[v])
    crit = A.critical_tolerance(adj, [hub], [0.0])           # Sweep besteht nur aus alpha=0 - fast sicher noch eine Kaskade
    rounds0, _ = A.motter_lai_cascade(adj, 0.0, [hub])
    if len(rounds0) > 1:
        assert crit is None
    else:
        assert crit == 0.0


def test_critical_tolerance_monotonic_growth_of_alpha_reduces_or_keeps_damage():
    """Nicht Teil der 10-Punkte-Kette, aber eine direkte Konsequenz: mit steigendem alpha kann der Gesamtausfallanteil G nur fallen oder gleich bleiben, nie steigen (mehr Toleranz kann nie mehr
    Sekundaerausfaelle ausloesen)."""
    inst = S.barabasi_albert_instance(50, 2, 4, 5)
    adj = A.adjacency(inst.n, inst.edges)
    node_between0, _, _ = A.betweenness_brandes(adj)
    hub = max(range(inst.n), key=lambda v: node_between0[v])
    alphas = [0.0, 0.2, 0.4, 0.6, 1.0, 2.0, 5.0]
    gs = [A.motter_lai_cascade(adj, a, [hub])[1] for a in alphas]
    assert all(gs[i] >= gs[i + 1] - 1e-9 for i in range(len(gs) - 1))


def test_invalid_initial_removed_raises():
    inst = S.barbell_instance(6)
    adj = A.adjacency(inst.n, inst.edges)
    with pytest.raises(ValueError):
        A.motter_lai_cascade(adj, 0.5, [999])
