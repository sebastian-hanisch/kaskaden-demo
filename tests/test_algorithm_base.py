"""Rauchtests der wortgleich kopierten Bausteine (adjacency, bfs_distances, degree_centrality, betweenness_brandes, components) gegen networkx als Gegenprobe."""

import networkx as nx
import pytest

import kas_algorithm as A
import kas_scenario as S


def _nx_graph(adj):
    g = nx.Graph()
    g.add_nodes_from(range(len(adj)))
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if u < v:
                g.add_edge(u, v)
    return g


INSTANCES = [S.generate(side=6, blocked=0.2, nettype="grid", seed=s) for s in range(1, 4)] + \
    [S.barabasi_albert_instance(30, 2, 4, s) for s in range(1, 4)] + \
    [S.barbell_instance(k) for k in (4, 6, 8)]


@pytest.mark.parametrize("inst", INSTANCES)
def test_betweenness_matches_networkx_unnormalized(inst):
    adj = A.adjacency(inst.n, inst.edges)
    node_between, _, _ = A.betweenness_brandes(adj)
    g = _nx_graph(adj)
    want = nx.betweenness_centrality(g, normalized=False)
    for v in range(inst.n):
        assert node_between[v] == pytest.approx(want[v], abs=1e-6)


@pytest.mark.parametrize("inst", INSTANCES)
def test_degree_centrality_matches_adjacency_length(inst):
    adj = A.adjacency(inst.n, inst.edges)
    deg = A.degree_centrality(adj)
    assert deg == [len(a) for a in adj]


@pytest.mark.parametrize("inst", INSTANCES)
def test_components_matches_networkx(inst):
    adj = A.adjacency(inst.n, inst.edges)
    sizes = sorted(A.components(adj))
    g = _nx_graph(adj)
    want = sorted(len(c) for c in nx.connected_components(g))
    assert sizes == want


def test_bfs_distances_matches_networkx_shortest_path_length():
    inst = S.generate(side=6, blocked=0.2, nettype="grid", seed=2)
    adj = A.adjacency(inst.n, inst.edges)
    g = _nx_graph(adj)
    for s in range(0, inst.n, 5):
        dist, _ = A.bfs_distances(adj, s)
        want = nx.single_source_shortest_path_length(g, s)
        for v in range(inst.n):
            if v in want:
                assert dist[v] == want[v]
            else:
                assert dist[v] == -1


def test_isolated_node_has_zero_betweenness_and_does_not_break_computation():
    n = 5
    edges = [(0, 1, 1.0), (1, 2, 1.0)]                    # nodes 3, 4 isolated
    adj = A.adjacency(n, edges)
    node_between, _, _ = A.betweenness_brandes(adj)
    assert node_between[3] == 0.0 and node_between[4] == 0.0
    assert node_between[1] == pytest.approx(1.0)          # node 1 lies on the only shortest path 0-1-2
