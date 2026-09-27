"""Rauchtests der drei Instanzen (Betriebsnetz, Skalenfreies Netz, Barbell) und von erdos_renyi_gnm. Wortgleiche Kopie aus robustheit-demo/tests/test_scenario.py."""

import networkx as nx
import pytest

import kas_constants as C
import kas_scenario as S


def _graph(inst):
    g = nx.Graph()
    g.add_nodes_from(range(inst.n))
    g.add_edges_from((u, v) for u, v, _ in inst.edges)
    return g


def test_city_grid_shape_and_connectivity():
    inst = S.generate(side=6, blocked=0.0, nettype="grid", seed=1)
    assert inst.n == 36
    assert inst.kind == "city" and inst.nettype == "grid"
    g = _graph(inst)
    assert nx.is_connected(g)


def test_city_random_same_edge_count_as_grid():
    grid = S.generate(side=6, blocked=0.2, nettype="grid", seed=3)
    rnd = S.generate(side=6, blocked=0.2, nettype="random", seed=3)
    assert grid.m == rnd.m
    assert rnd.nettype == "random"


def test_city_invalid_side_raises():
    with pytest.raises(ValueError):
        S.generate(side=1)
    with pytest.raises(ValueError):
        S.generate(side=6, nettype="hexagonal")


def test_barabasi_albert_exact_edge_count_connected():
    for n, m, m0, seed in [(20, 2, 4, 1), (50, 1, 5, 2), (80, 3, 6, 3)]:
        inst = S.barabasi_albert_instance(n, m, m0, seed)
        assert inst.n == n
        assert inst.m == m0 + (n - m0) * m
        g = _graph(inst)
        assert nx.is_connected(g)


def test_barabasi_albert_invalid_params_raise():
    with pytest.raises(ValueError):
        S.barabasi_albert_instance(20, 2, 2, 1)          # m0 < 3
    with pytest.raises(ValueError):
        S.barabasi_albert_instance(20, 5, 4, 1)          # m > m0
    with pytest.raises(ValueError):
        S.barabasi_albert_instance(3, 1, 4, 1)           # n < m0


def test_barbell_degrees_and_bridge():
    inst = S.barbell_instance(k=5)
    assert inst.n == 10
    adj = {u: set() for u in range(inst.n)}
    for u, v, _ in inst.edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(4):                    # clique nodes without bridge end (0-3 left, 6-9 right): degree k-1 = 4
        assert len(adj[i]) == 4
    for i in range(6, 10):
        assert len(adj[i]) == 4
    assert len(adj[4]) == 5               # bridge end left: degree k = 5
    assert len(adj[5]) == 5               # bridge end right: degree k = 5
    pairs = [(u, v) for u, v, _ in inst.edges]
    assert (4, 5) in pairs


def test_barbell_invalid_k_raises():
    with pytest.raises(ValueError):
        S.barbell_instance(k=2)
    with pytest.raises(ValueError):
        S.barbell_instance(k=C.BARBELL_K_MAX + 1)


def test_erdos_renyi_exact_edge_count_no_self_loops_no_multi():
    for n in (5, 10, 20, 50):
        for m in (3, 10, 25):
            max_m = n * (n - 1) // 2
            if m > max_m:
                continue
            for seed in range(1, 4):
                pairs = S.erdos_renyi_gnm(n, m, seed)
                assert len(pairs) == m
                assert len(set(pairs)) == m
                assert all(u != v for u, v in pairs)
                assert all(u < v for u, v in pairs)


def test_determinism():
    a = S.generate(side=6, blocked=0.2, nettype="grid", seed=7)
    b = S.generate(side=6, blocked=0.2, nettype="grid", seed=7)
    assert a.edges == b.edges
    assert S.erdos_renyi_gnm(20, 30, seed=1) == S.erdos_renyi_gnm(20, 30, seed=1)
