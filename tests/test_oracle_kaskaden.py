"""Unabhängiges Orakel für Kaskade und SIR: (1) Motter-Lai-Kaskade gegen eine EXAKT rationale Neuberechnung (Fractions, Pfadzählung) auf kleinen und symmetrischen Graphen; (2) Startknoten
"höchste Last" mit der Gleichstandsregel kleinster Index gegen exakte Betweenness (früher entschied bei symmetrischen Netzen, z. B. dem ungesperrten Raster, Gleitkommarauschen); (3) SIR gegen die
EXAKTE Endgrößenverteilung (Markov-Kette über alle S/I/R-Zustände) mit Monte-Carlo-Band; (4) Epidemieschwelle als geschlossene Formel."""

import itertools
import math
from collections import Counter, deque
from fractions import Fraction
from functools import lru_cache

import networkx as nx
import numpy as np
import pytest

import kas_algorithm as A
import kas_evaluation as ev
import kas_scenario as S


def _bw_exact(nb, alive):
    n = len(nb)
    bc = [Fraction(0)] * n
    for s in range(n):
        if not alive[s]:
            continue
        dist, sigma, pred, order, q = {s: 0}, {s: 1}, {s: []}, [], deque([s])
        while q:
            u = q.popleft()
            order.append(u)
            for v in nb[u]:
                if not alive[v]:
                    continue
                if v not in dist:
                    dist[v], sigma[v], pred[v] = dist[u] + 1, 0, []
                    q.append(v)
                if dist[v] == dist[u] + 1:
                    sigma[v] += sigma[u]
                    pred[v].append(u)
        delta = {v: Fraction(0) for v in order}
        for w in reversed(order):
            for v in pred[w]:
                delta[v] += Fraction(sigma[v], sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    return [x / 2 for x in bc]


def _exact_cascade(adj, alpha, init):
    n = len(adj)
    cap = [(1 + Fraction(alpha)) * b for b in _bw_exact(adj, [True] * n)]
    alive = [True] * n
    for v in init:
        alive[v] = False
    rounds = [sorted(init)]
    while True:
        bc = _bw_exact(adj, alive)
        failed = [v for v in range(n) if alive[v] and bc[v] > cap[v]]
        if not failed:
            return rounds
        for v in failed:
            alive[v] = False
        rounds.append(failed)


def _graphs():
    gs = [nx.cycle_graph(6), nx.hypercube_graph(3), nx.petersen_graph(), nx.barbell_graph(3, 0), nx.barbell_graph(4, 0), nx.complete_bipartite_graph(3, 3), nx.wheel_graph(7),
          nx.grid_2d_graph(3, 3), nx.circulant_graph(9, [1, 2]), nx.path_graph(7), nx.star_graph(5), nx.erdos_renyi_graph(9, 0.35, seed=1), nx.erdos_renyi_graph(10, 0.3, seed=2)]
    out = []
    for g in gs:
        g = nx.convert_node_labels_to_integers(g)
        out.append((g.number_of_nodes(), sorted((min(u, v), max(u, v)) for u, v in g.edges)))
    return out


def test_cascade_matches_exact_rational_recomputation():
    for n, pairs in _graphs():
        adj = A.adjacency(n, [(u, v, 1.0) for u, v in pairs])
        for alpha in (0.0, 0.25, 0.5, 1.0, 2.0):
            for v in range(n):
                rounds, G = A.motter_lai_cascade(adj, alpha, [v])
                want = _exact_cascade(adj, alpha, [v])
                assert [r["newly_failed"] for r in rounds] == want
                assert G == pytest.approx(sum(len(r) for r in want) / n)


@pytest.mark.parametrize("side", [4, 8, 10])
def test_highest_load_start_node_breaks_exact_ties_by_smallest_index(side):
    inst = S.generate(side, 0.0, "grid", 1)                                  # ungesperrtes Raster: 4 bzw. 1 mathematisch gleiche Mittelknoten
    adj = A.adjacency(inst.n, inst.edges)
    bc = _bw_exact(adj, [True] * inst.n)
    assert ev.initial_node(adj, "highest_load", 0) == max(range(inst.n), key=lambda v: (bc[v], -v))


def _exact_sir_size_distribution(n, nbrs, beta, mu, p0):
    @lru_cache(None)
    def dist(state):
        inf = [v for v in range(n) if state[v] == 1]
        if not inf:
            return {n - state.count(0): 1.0}
        sus = [u for u in range(n) if state[u] == 0]
        pinf = {u: 1 - (1 - beta) ** sum(1 for w in nbrs[u] if state[w] == 1) for u in sus}
        out = {}
        for infm in itertools.product((0, 1), repeat=len(sus)):
            pr = 1.0
            for u, b in zip(sus, infm):
                pr *= pinf[u] if b else 1 - pinf[u]
            if pr == 0:
                continue
            for recm in itertools.product((0, 1), repeat=len(inf)):
                p2 = pr
                for v, b in zip(inf, recm):
                    p2 *= mu if b else 1 - mu
                if p2 == 0:
                    continue
                new = list(state)
                for u, b in zip(sus, infm):
                    if b:
                        new[u] = 1
                for v, b in zip(inf, recm):
                    if b:
                        new[v] = 2
                out[tuple(new)] = out.get(tuple(new), 0.0) + p2
        stay = out.pop(state, 0.0)
        if stay > 1 - 1e-12:
            return {n - state.count(0): 1.0}
        res = {}
        for st, p in out.items():
            for k, q in dist(st).items():
                res[k] = res.get(k, 0.0) + p * q / (1 - stay)
        return res

    st = [0] * n
    st[p0] = 1
    return dist(tuple(st))


@pytest.mark.parametrize("beta,mu", [(0.3, 0.5), (0.6, 0.2), (0.4, 0.0), (0.15, 0.05)])
def test_sir_outbreak_size_distribution_matches_exact_markov_chain(beta, mu):
    cases = [(4, [(0, 1), (1, 2), (2, 3)]), (4, [(0, 1), (1, 2), (2, 0), (2, 3)]), (5, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)])]
    runs = 2000
    for idx, (n, pairs) in enumerate(cases):
        adj = A.adjacency(n, [(u, v, 1.0) for u, v in pairs])
        exact = _exact_sir_size_distribution(n, adj, beta, mu, 0)
        mean = sum(k * p for k, p in exact.items())
        var = sum((k - mean) ** 2 * p for k, p in exact.items())
        stats = A.sir_many_runs(adj, beta, mu, runs, 100 + idx, 0)
        assert abs(stats["mean"] - mean) < 5 * math.sqrt(var / runs) + 1e-9
        cnt = Counter(stats["runs"])
        big = [k for k, p in exact.items() if runs * p >= 5]
        chi = sum((cnt.get(k, 0) - runs * exact[k]) ** 2 / (runs * exact[k]) for k in big)
        assert chi < len(big) + 6 * math.sqrt(2 * len(big)) + 10


def test_epidemic_threshold_closed_form_from_measured_moments():
    rng = np.random.default_rng(3)
    for _ in range(60):
        n = int(rng.integers(2, 12))
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.4]
        adj = A.adjacency(n, [(u, v, 1.0) for u, v in pairs])
        deg = np.array([len(a) for a in adj], dtype=float)
        got = A.epidemic_threshold_prediction(adj, 0.3)
        if (deg ** 2).mean() > 0:
            assert got == pytest.approx(0.3 * deg.mean() / (deg ** 2).mean())
        else:
            assert got is None
