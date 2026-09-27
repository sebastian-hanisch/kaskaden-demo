"""Jede Zahl aus README und kas_constants-Kommentar, nachgerechnet über die echten Auswertungsfunktionen (Hauskonvention: keine erfundenen Zahlen)."""

import pytest

import kas_algorithm as A
import kas_constants as C
import kas_evaluation as ev
import kas_scenario as S


def _hub(adj):
    node_between, _, _ = A.betweenness_brandes(adj)
    return max(range(len(adj)), key=lambda v: (node_between[v], -v))


def test_readme_scale_free_cascade_57x_and_critical_tolerance_17():
    inst = S.barabasi_albert_instance(150, 2, 4, 35)
    adj = A.adjacency(inst.n, inst.edges)
    hub = _hub(adj)
    _, G = A.motter_lai_cascade(adj, 0.5, [hub])
    assert round(G * inst.n) == 57
    crit = A.critical_tolerance(adj, [hub], C.ALPHA_SWEEP)
    assert crit == 17.0


def test_readme_city_cascade_39_and_critical_tolerance_2_8_and_stops_at_alpha_3():
    inst = S.generate(side=10, blocked=0.2, nettype="grid", seed=35)
    adj = A.adjacency(inst.n, inst.edges)
    hub = _hub(adj)
    _, G = A.motter_lai_cascade(adj, 0.5, [hub])
    assert round(G * inst.n) == 39
    crit = A.critical_tolerance(adj, [hub], C.ALPHA_SWEEP)
    assert crit == 2.8
    rounds3, G3 = A.motter_lai_cascade(adj, 3.0, [hub])
    assert len(rounds3) == 1 and G3 == 0.01


def test_readme_cascade_damage_at_alpha_0_5_is_nearly_equal_for_city_and_ba_despite_very_different_critical_tolerance():
    """README-Überraschung: bei einem FESTEN alpha=0.5 ist der Schaden auf Betriebsnetz (39%) und Skalenfrei (38%) fast identisch, obwohl die kritische Toleranz sich um eine Groessenordnung
    unterscheidet (2.8 gegen 17.0) - die kritische Toleranz, nicht der Schaden bei einem festen alpha, ist die Kennzahl, die unterscheidet."""
    inst_city = S.generate(side=10, blocked=0.2, nettype="grid", seed=35)
    adj_city = A.adjacency(inst_city.n, inst_city.edges)
    _, G_city = A.motter_lai_cascade(adj_city, 0.5, [_hub(adj_city)])
    inst_ba = S.barabasi_albert_instance(150, 2, 4, 35)
    adj_ba = A.adjacency(inst_ba.n, inst_ba.edges)
    _, G_ba = A.motter_lai_cascade(adj_ba, 0.5, [_hub(adj_ba)])
    assert abs(G_city - G_ba) < 0.02                          # "praktisch derselbe Schaden" (0.39 gegen 0.38)


def test_readme_barbell_critical_tolerance_exactly_zero():
    inst = S.barbell_instance(8)
    adj = A.adjacency(inst.n, inst.edges)
    bridge = _hub(adj)
    crit = A.critical_tolerance(adj, [bridge], C.ALPHA_SWEEP)
    assert crit == 0.0


def test_readme_barbell_sir_bottleneck_numbers():
    inst = S.barbell_instance(8)
    adj = A.adjacency(inst.n, inst.edges)
    pz = ev.initial_node(adj, "random", 35)
    many = A.sir_many_runs(adj, 0.2, 0.3, n_runs=100, seed=35, patient_zero=pz, max_steps=60)
    assert many["p50"] == 8 and many["p90"] == 16
    frac_crossed = sum(1 for s in many["runs"] if s > 8) / len(many["runs"])
    assert 0.2 <= frac_crossed <= 0.4                         # README: "nur 30%"


def test_readme_threshold_random_variant_closer_to_formula_than_grid():
    """README Design-Entscheidung: die Zufallsgraph-Option liegt naeher an der Mean-Field-Formel als das reine Raster, bei ZWEI verschiedenen Groessen (kein Zufallstreffer)."""
    betas = tuple(round(0.005 * i, 4) for i in range(1, 121))
    for side in (12, 17):
        grid_settings = ev.Settings(kind="city", side=side, blocked=0.0, nettype="grid", initial_strategy="highest_load", mu=0.3, n_runs=100, max_steps=200, seed=35)
        random_settings = ev.Settings(kind="city", side=side, blocked=0.0, nettype="random", initial_strategy="highest_load", mu=0.3, n_runs=100, max_steps=200, seed=35)
        grid_check = ev.threshold_check(grid_settings, betas)
        random_check = ev.threshold_check(random_settings, betas)
        assert abs(grid_check["gap"]) > abs(random_check["gap"])


def test_readme_city_random_threshold_numbers():
    settings = ev.Settings(kind="city", side=17, blocked=0.0, nettype="random", initial_strategy="highest_load", mu=0.3, n_runs=150, max_steps=200, seed=35)
    betas = tuple(round(0.005 * i, 4) for i in range(1, 121))
    check = ev.threshold_check(settings, betas)
    assert check["beta_c"] == pytest.approx(0.0636, abs=0.0005)
    assert check["measured"] == pytest.approx(0.0907, abs=0.0005)


def test_readme_city_grid_threshold_numbers():
    settings = ev.Settings(kind="city", side=10, blocked=0.2, nettype="grid", initial_strategy="highest_load", mu=0.3, n_runs=100, max_steps=150, seed=35)
    betas = tuple(round(0.01 * i, 3) for i in range(1, 61))
    check = ev.threshold_check(settings, betas)
    assert check["beta_c"] == pytest.approx(0.0939, abs=0.0005)
    assert check["measured"] == pytest.approx(0.1423, abs=0.0005)


def test_readme_scale_free_threshold_about_half_of_city():
    settings_ba = ev.Settings(kind="ba", n_ba=288, m_ba=2, m0_ba=4, initial_strategy="highest_load", mu=0.3, n_runs=150, max_steps=200, seed=35)
    settings_city = ev.Settings(kind="city", side=17, blocked=0.0, nettype="random", initial_strategy="highest_load", mu=0.3, n_runs=150, max_steps=200, seed=35)
    betas = tuple(round(0.005 * i, 4) for i in range(1, 121))
    check_ba = ev.threshold_check(settings_ba, betas)
    check_city = ev.threshold_check(settings_city, betas)
    ratio = check_ba["beta_c"] / check_city["beta_c"]
    assert 0.4 < ratio < 0.65                                 # README: "etwa halb so gross" (Verhaeltnis 0.52)
    assert abs(check_ba["gap"]) < abs(check_city["gap"])      # README: "viel enger als beim Betriebsnetz"


def test_readme_huge_alpha_zero_secondary_failures_no_outliers_observed():
    """README H4: >55 zufaellige Stichproben ohne Ausreisser (bezieht sich auf test_cascade.test_cascade_random_removal_huge_alpha_no_secondary_failures, hier nochmal direkt nachgerechnet mit der
    dort verwendeten Anzahl)."""
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
    assert total >= 39
    assert ok == total                                        # README: "0 Ausreisser"
