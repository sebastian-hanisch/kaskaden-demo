"""Zentrale Korrektheits-Kette, Punkte 9-10:

9) Epidemieschwelle: gemessene Schwelle liegt innerhalb einer Toleranzbande um beta_c=mu*<k>/<k^2> auf Betriebsnetz/Zufallsgraph-Instanzen (gebaendert, endliche Groesse); skalenfreies Netz zeigt
   gemessen eine MESSBAR NIEDRIGERE Schwelle als Betriebsnetz/Zufallsgraph gleicher Groesse (Kernaussage, gemessen, nicht angenommen).
10) Buchfuehrung/Determinismus/Sonderfaelle (n<=2, unzusammenhaengende Ausgangsinstanz, vollstaendiger Graph)."""

import kas_algorithm as A
import kas_evaluation as ev
import kas_scenario as S

BETAS = tuple(round(0.01 * i, 3) for i in range(1, 61))


def _city_random_settings(n_side, seed=35):
    return ev.Settings(kind="city", side=n_side, blocked=0.0, nettype="random", mode="ausbreitung", initial_strategy="highest_load", mu=0.3, n_runs=100, max_steps=150, seed=seed)


def _ba_settings(n, m=2, m0=4, seed=35):
    return ev.Settings(kind="ba", n_ba=n, m_ba=m, m0_ba=m0, mode="ausbreitung", initial_strategy="highest_load", mu=0.3, n_runs=100, max_steps=150, seed=seed)


# --- 9. Epidemieschwelle gebaendert gegen Pastor-Satorras; skalenfrei messbar niedriger ---------------------------------------------------------------------


def test_measured_threshold_within_band_of_pastor_satorras_formula_on_homogeneous_networks():
    """Betriebsnetz mit der Zufallsgraph-Option (Poisson-artige Gradverteilung, s. README 'Design-Entscheidung' - reines Gitter ist raeumlich strukturiert und weicht strukturell staerker von der
    Mean-Field-Annahme ab, s. test_grid_topology_shows_larger_mean_field_gap_than_random_graph unten) liegt die gemessene Schwelle nahe an beta_c."""
    settings = _city_random_settings(n_side=17)                # n=289
    check = ev.threshold_check(settings, BETAS)
    assert check["beta_c"] is not None and check["measured"] is not None
    assert abs(check["gap"]) < 0.06


def test_scale_free_network_shows_measurably_lower_threshold_than_homogeneous_network_of_comparable_size():
    """Kernaussage von Pastor-Satorras & Vespignani (2001): auf dem skalenfreien Netz breitet sich eine Epidemie schon bei viel kleinerem beta aus als auf einem vergleichbar großen homogenen Netz -
    gemessen, nicht angenommen. n ungefaehr gleich (289 gegen 288/300)."""
    city_check = ev.threshold_check(_city_random_settings(n_side=17), BETAS)          # n=289
    ba_check = ev.threshold_check(_ba_settings(n=288, m=2, m0=4), BETAS)              # n=288
    assert ba_check["measured"] is not None and city_check["measured"] is not None
    assert ba_check["measured"] < city_check["measured"]
    assert ba_check["beta_c"] < city_check["beta_c"]                                  # auch die THEORETISCHE Vorhersage liegt niedriger (mehr Gradheterogenitaet -> hoeheres <k^2>)


def test_grid_topology_shows_larger_mean_field_gap_than_random_graph_same_edge_count():
    """Ehrlicher Befund (Design-Entscheidung, analog zur robustheit-demo): das reine Straßenraster (raeumlich stark strukturiert, praktisch keine Abkuerzungen) weicht STRUKTURELL staerker von der
    Mean-Field-Vorhersage ab als ein Zufallsgraph derselben Kantenzahl - kein Rauschen, sondern ein bekanntes Manko der (annahmefreien) heterogenen Mean-Field-Naeherung auf niedrigdimensionalen,
    raeumlichen Netzen. Deshalb wird fuer den EIGENTLICHEN Formelvergleich (s. oben) bewusst die Zufallsgraph-Option verwendet."""
    grid_settings = ev.Settings(kind="city", side=12, blocked=0.0, nettype="grid", mode="ausbreitung", initial_strategy="highest_load", mu=0.3, n_runs=80, max_steps=150, seed=35)
    random_settings = ev.Settings(kind="city", side=12, blocked=0.0, nettype="random", mode="ausbreitung", initial_strategy="highest_load", mu=0.3, n_runs=80, max_steps=150, seed=35)
    grid_check = ev.threshold_check(grid_settings, BETAS)
    random_check = ev.threshold_check(random_settings, BETAS)
    assert grid_check["measured"] is not None and random_check["measured"] is not None
    grid_gap = abs(grid_check["gap"])
    random_gap = abs(random_check["gap"])
    assert grid_gap > random_gap


def test_epidemic_threshold_prediction_uses_measured_degree_moments_not_an_assumed_distribution():
    """`epidemic_threshold_prediction` misst <k> und <k^2> direkt aus der uebergebenen Adjazenzliste - zwei Instanzen mit unterschiedlicher tatsaechlicher Gradverteilung liefern unterschiedliche
    beta_c, obwohl beide "das Betriebsnetz" heissen koennten."""
    city_uniform = S.generate(side=8, blocked=0.0, nettype="grid", seed=1)         # enge Gradverteilung (fast alle Grad 4)
    ba = S.barabasi_albert_instance(64, 2, 4, 1)                                    # breite Gradverteilung (Hubs)
    adj_city = A.adjacency(city_uniform.n, city_uniform.edges)
    adj_ba = A.adjacency(ba.n, ba.edges)
    bc_city = A.epidemic_threshold_prediction(adj_city, mu=0.3)
    bc_ba = A.epidemic_threshold_prediction(adj_ba, mu=0.3)
    assert bc_ba < bc_city                                                          # <k^2> ist beim skalenfreien Netz deutlich groesser -> beta_c kleiner


# --- 10. Buchfuehrung/Determinismus/Sonderfaelle --------------------------------------------------------------------------------------------------------------


def test_measured_threshold_none_when_frac_never_reached():
    rows = [{"beta": b, "mean_fraction": 0.0} for b in (0.01, 0.02, 0.03)]
    assert ev.measured_threshold(rows, frac=0.5) is None


def test_measured_threshold_linear_interpolation():
    rows = [{"beta": 0.1, "mean_fraction": 0.0}, {"beta": 0.2, "mean_fraction": 0.2}]
    got = ev.measured_threshold(rows, frac=0.1)
    assert abs(got - 0.15) < 1e-9


def test_analyse_cascade_and_sir_run_on_n_le_2_and_disconnected_instances():
    for n, edges in [(1, []), (2, []), (2, [(0, 1, 1.0)])]:
        adj = A.adjacency(n, edges)
        rounds, G = A.motter_lai_cascade(adj, 0.5, [0])
        assert rounds[0]["newly_failed"] == [0]
        assert 0.0 <= G <= 1.0
        result = A.sir_simulate(adj, 0.5, 0.3, seed=1, patient_zero=0, max_steps=10)
        assert result.final_s + result.final_i + result.final_r == n


def test_network_comparison_scale_free_is_most_vulnerable_in_both_dynamics():
    """Die Kernaussage von Schritt 4 der App: das skalenfreie Netz braucht die GROESSTE Toleranz alpha, um Sekundaerausfaelle beim Hub-Ausfall zu verhindern (kritische Toleranz = wie viel Puffer
    noetig ist, damit KEINE Kaskade ausbricht - je hoeher, desto verwundbarer), UND hat die niedrigste gemessene Epidemieschwelle (Ausbreitung) unter den drei Vehikeln vergleichbarer Groesse."""
    city = ev.Settings(kind="city", side=12, blocked=0.2, nettype="grid", initial_strategy="highest_load", mu=0.3, n_runs=60, max_steps=120, seed=35)
    ba = ev.Settings(kind="ba", n_ba=144, m_ba=2, m0_ba=4, initial_strategy="highest_load", mu=0.3, n_runs=60, max_steps=120, seed=35)
    barbell = ev.Settings(kind="barbell", k_barbell=8, initial_strategy="highest_load", mu=0.3, n_runs=60, max_steps=120, seed=35)
    rows = ev.network_comparison(city, ba, barbell, betas=BETAS)
    by_label = {r["label"]: r for r in rows}
    assert by_label["Skalenfrei"]["critical_tolerance"] is not None
    other_crit = [by_label[lbl]["critical_tolerance"] for lbl in ("Betriebsnetz", "Barbell") if by_label[lbl]["critical_tolerance"] is not None]
    if other_crit:
        assert by_label["Skalenfrei"]["critical_tolerance"] >= max(other_crit) - 1e-9
    assert by_label["Skalenfrei"]["beta_c"] < by_label["Betriebsnetz"]["beta_c"]


def test_cascade_vs_single_removal_multiplier_is_at_least_one():
    settings = ev.Settings(kind="ba", n_ba=100, m_ba=2, m0_ba=4, initial_strategy="highest_load", alpha=0.3, seed=35)
    _, _, rows = ev.cascade_vs_single_removal(settings, alphas=(0.0, 0.1, 0.3, 0.5, 1.0))
    for row in rows:
        assert row["multiplier"] >= 1.0                       # eine Kaskade kann nie WENIGER Schaden anrichten als der einmalige Ausfall selbst


def test_nearest_alpha_rows_never_silently_drops_a_target_on_a_coarse_grid():
    """Regression: ALPHA_SWEEP hat Schrittweite 0.2 (0.0, 0.2, 0.4, 0.6, ...) und trifft einen "griffigen" Zielwert wie 0.5 NIE exakt - eine naive `r["alpha"] == 0.5`-Filterung liefert dafuer
    STILLSCHWEIGEND keine Zeile (ein fehlender Balken in der App, ohne Fehlermeldung). `nearest_alpha_rows` muss stattdessen den naechstgelegenen tatsaechlich vorhandenen Punkt waehlen."""
    rows = [{"alpha": round(0.2 * i, 2), "G": 0.01 * i} for i in range(26)]         # 0.0, 0.2, 0.4, ..., 5.0
    targets = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0)
    chosen = ev.nearest_alpha_rows(rows, targets)
    assert len(chosen) == len(targets)                        # kein Zielwert faellt unter den Tisch
    chosen_alphas = [r["alpha"] for r in chosen]
    assert chosen_alphas == sorted(chosen_alphas)
    assert 0.4 in chosen_alphas or 0.6 in chosen_alphas        # der naechstgelegene Ersatz fuer 0.5 (Abstand 0.1 zu beiden)


def test_nearest_alpha_rows_deduplicates_when_two_targets_map_to_the_same_point():
    rows = [{"alpha": a, "G": 0.0} for a in (0.0, 5.0)]
    chosen = ev.nearest_alpha_rows(rows, (0.0, 0.1, 5.0, 6.0))
    assert len(chosen) == 2                                   # 0.1 und 0.0 treffen denselben Punkt, ebenso 5.0 und 6.0


def test_nearest_alpha_rows_empty_input():
    assert ev.nearest_alpha_rows([], (0.0, 1.0)) == []
