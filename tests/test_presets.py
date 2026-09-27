"""Jede Zahl in den PRESET_HELP-Hilfetexten, nachgerechnet über die echten Auswertungsfunktionen (Hauskonvention: keine erfundenen Zahlen)."""

import kas_algorithm as A
import kas_constants as C
import kas_evaluation as ev
import kas_scenario as S


def _hub(adj):
    node_between, _, _ = A.betweenness_brandes(adj)
    return max(range(len(adj)), key=lambda v: (node_between[v], -v))


def test_barbell_kaskade_preset_numbers():
    inst = S.barbell_instance(8)
    adj = A.adjacency(inst.n, inst.edges)
    bridge = _hub(adj)
    rounds, G = A.motter_lai_cascade(adj, 0.5, [bridge])
    assert len(rounds) == 1
    assert G == 1.0 / 16


def test_barbell_ausbreitung_preset_numbers():
    inst = S.barbell_instance(8)
    adj = A.adjacency(inst.n, inst.edges)
    pz = ev.initial_node(adj, "random", 35)
    many = A.sir_many_runs(adj, 0.2, 0.3, n_runs=100, seed=35, patient_zero=pz, max_steps=60)
    frac_crossed = sum(1 for s in many["runs"] if s > 8) / len(many["runs"])
    assert many["p50"] == 8
    assert many["p90"] == 16
    assert 0.2 <= frac_crossed <= 0.4                       # README: "nur 30%"


def test_skalenfrei_kaskade_preset_numbers():
    inst = S.barabasi_albert_instance(150, 2, 4, 35)
    adj = A.adjacency(inst.n, inst.edges)
    hub = _hub(adj)
    rounds, G = A.motter_lai_cascade(adj, C.DEFAULT_ALPHA, [hub])
    assert round(G * inst.n) == 57
    assert G == 0.38
    crit = A.critical_tolerance(adj, [hub], C.ALPHA_SWEEP)
    assert crit == 17.0


def test_kritische_toleranz_barbell_exact_zero_preset_numbers():
    inst = S.barbell_instance(8)
    adj = A.adjacency(inst.n, inst.edges)
    bridge = _hub(adj)
    crit = A.critical_tolerance(adj, [bridge], C.ALPHA_SWEEP)
    assert crit == 0.0


def test_betriebsnetz_robust_preset_numbers():
    inst = S.generate(side=10, blocked=0.2, nettype="grid", seed=35)
    adj = A.adjacency(inst.n, inst.edges)
    hub = _hub(adj)
    crit = A.critical_tolerance(adj, [hub], C.ALPHA_SWEEP)
    assert crit == 2.8
    rounds, G = A.motter_lai_cascade(adj, 3.0, [hub])
    assert len(rounds) == 1
    assert G == 0.01


def test_epidemieschwelle_betriebsnetz_preset_numbers():
    settings = ev.Settings(kind="city", side=17, blocked=0.0, nettype="random", initial_strategy="highest_load", mu=0.3, n_runs=150, max_steps=200, seed=35)
    betas = tuple(round(0.005 * i, 4) for i in range(1, 121))
    check = ev.threshold_check(settings, betas)
    assert check["inst"].n == 289
    assert check["beta_c"] == _approx(0.0636)
    assert check["measured"] == _approx(0.0907)
    assert 0.0 < check["gap"] < 0.06


def test_skalenfrei_threshold_preset_numbers():
    settings = ev.Settings(kind="ba", n_ba=288, m_ba=2, m0_ba=4, initial_strategy="highest_load", mu=0.3, n_runs=150, max_steps=200, seed=35)
    betas = tuple(round(0.005 * i, 4) for i in range(1, 121))
    check = ev.threshold_check(settings, betas)
    assert check["inst"].n == 288
    assert check["beta_c"] == _approx(0.0330)
    assert check["measured"] == _approx(0.0397)


def test_drei_netze_vergleich_preset_numbers():
    city = ev.Settings(kind="city", side=10, blocked=0.2, nettype="grid", initial_strategy="highest_load", mu=0.3, n_runs=100, max_steps=150, seed=35)
    ba = ev.Settings(kind="ba", n_ba=150, m_ba=2, m0_ba=4, initial_strategy="highest_load", mu=0.3, n_runs=100, max_steps=150, seed=35)
    barbell = ev.Settings(kind="barbell", k_barbell=8, initial_strategy="highest_load", mu=0.3, n_runs=100, max_steps=150, seed=35)
    rows = ev.network_comparison(city, ba, barbell)
    by_label = {r["label"]: r for r in rows}
    assert by_label["Betriebsnetz"]["critical_tolerance"] == 2.8
    assert by_label["Skalenfrei"]["critical_tolerance"] == 17.0
    assert by_label["Barbell"]["critical_tolerance"] == 0.0
    assert by_label["Skalenfrei"]["beta_c"] < by_label["Betriebsnetz"]["beta_c"]
    assert by_label["Skalenfrei"]["beta_c"] < by_label["Barbell"]["beta_c"]


def _approx(x, abs_tol=0.0005):
    import pytest
    return pytest.approx(x, abs=abs_tol)
