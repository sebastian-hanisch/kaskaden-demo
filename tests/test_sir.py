"""Zentrale Korrektheits-Kette, Punkte 5-8 (Pastor-Satorras & Vespignani 2001, SIR-Ausbreitung):

5) S+I+R = n exakt in JEDEM Zeitschritt (Erhaltungssatz, Test in jedem Schritt, nicht nur am Ende).
6) SIR bei beta=0: R(unendlich)=1 exakt (keine Ausbreitung ueber Patient Null hinaus, Endgroesse == 1).
7) SIR bei beta=1, mu=0: Endgroesse == Groesse der Zusammenhangskomponente von Patient Null EXAKT (deterministische BFS-Wellenfront, gegen `bfs_distances` geprueft, kein Band noetig).
8) Mittlere Endgroesse nicht fallend in beta (ueber viele Laeufe gemittelt, gebaendert) auf >= 100 Instanzen."""

import pytest

import kas_algorithm as A
import kas_scenario as S


def _instances():
    out = []
    for side in (4, 5, 6):
        for seed in (1, 2, 3):
            out.append(S.generate(side, 0.2, "grid", seed))
    for n, m, m0, seed in [(15, 2, 4, 1), (25, 2, 5, 2), (40, 1, 4, 3)]:
        out.append(S.barabasi_albert_instance(n, m, m0, seed))
    for k in (4, 6, 8):
        out.append(S.barbell_instance(k))
    return out


INSTANCES = _instances()


# --- 5. Erhaltungssatz: S+I+R = n in JEDEM Zeitschritt ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("inst", INSTANCES)
def test_sir_conservation_law_every_step(inst):
    adj = A.adjacency(inst.n, inst.edges)
    for seed in (1, 2, 3):
        result = A.sir_simulate(adj, beta=0.3, mu=0.2, seed=seed, patient_zero=0, max_steps=100)
        for s, i, r in result.trace:
            assert s + i + r == inst.n
        assert result.final_s + result.final_i + result.final_r == inst.n
        assert result.outbreak_size == inst.n - result.final_s


# --- 6. beta=0: keine Ausbreitung ueber Patient Null hinaus -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("inst", INSTANCES)
def test_sir_beta_zero_no_spread_beyond_patient_zero(inst):
    adj = A.adjacency(inst.n, inst.edges)
    for seed in (1, 2, 3, 4, 5):
        result = A.sir_simulate(adj, beta=0.0, mu=0.3, seed=seed, patient_zero=0, max_steps=50)
        assert result.outbreak_size == 1


def test_sir_beta_zero_mu_zero_patient_zero_stays_infected_forever():
    """Sonderfall beta=0, mu=0: Patient Null bleibt fuer immer infiziert (kein Ausbreitungs-, kein Genesungsereignis), Endgroesse trotzdem exakt 1."""
    inst = S.barabasi_albert_instance(20, 2, 4, 1)
    adj = A.adjacency(inst.n, inst.edges)
    result = A.sir_simulate(adj, beta=0.0, mu=0.0, seed=1, patient_zero=0, max_steps=30)
    assert result.outbreak_size == 1
    assert result.final_i == 1 and result.final_r == 0
    assert result.steps_run == 30                            # laeuft bis max_steps, da nie "keine Infizierten mehr"


# --- 7. beta=1, mu=0: Endgroesse == Groesse der Zusammenhangskomponente (deterministischer Grenzfall) ------------------------------------------------------


@pytest.mark.parametrize("inst", INSTANCES)
def test_sir_beta_one_mu_zero_matches_bfs_component_size_exactly(inst):
    """beta=1: jede Ansteckung gelingt garantiert; mu=0: niemand genest je - degeneriert zu einer deterministischen BFS-Wellenfront. Kein Zufall mehr im Spiel (der einzige rng-Aufruf,
    `rng.random() < beta`, ist mit beta=1.0 immer wahr; `rng.random() < mu` mit mu=0.0 immer falsch) - die Endgroesse MUSS exakt der Groesse der Zusammenhangskomponente von Patient Null
    entsprechen, kein Toleranzband noetig."""
    adj = A.adjacency(inst.n, inst.edges)
    dist, _ = A.bfs_distances(adj, 0)
    reachable = sum(1 for d in dist if d >= 0)
    max_steps = inst.n + 2                                    # BFS-Tiefe ist hoechstens n-1
    for seed in (1, 2, 3):
        result = A.sir_simulate(adj, beta=1.0, mu=0.0, seed=seed, patient_zero=0, max_steps=max_steps)
        assert result.outbreak_size == reachable
        assert result.final_r == 0                            # mu=0: niemand genest je


def test_sir_beta_one_mu_zero_reaches_whole_graph_when_connected():
    inst = S.generate(side=6, blocked=0.0, nettype="grid", seed=1)      # unblockiert -> zusammenhaengend
    adj = A.adjacency(inst.n, inst.edges)
    result = A.sir_simulate(adj, beta=1.0, mu=0.0, seed=1, patient_zero=0, max_steps=inst.n + 2)
    assert result.outbreak_size == inst.n


# --- 8. Mittlere Endgroesse nicht fallend in beta (gebaendert, >=100 Instanzen) -----------------------------------------------------------------------------


def test_mean_outbreak_size_non_decreasing_in_beta_over_many_instances():
    """Mittelwert-Aussage ueber >=100 (Instanz, Seed)-Kombinationen: ein hoeheres beta darf die MITTLERE Endgroesse (ueber viele Laeufe gemittelt) nie verringern - gebaendert (kleine
    Rueckwaertsabweichungen durch Stichprobenrauschen sind bei n_runs=40 je Punkt moeglich, s. Toleranz)."""
    checked = 0
    violations = 0
    betas = (0.05, 0.2, 0.5, 0.9)
    for n, m, m0 in [(20, 2, 4), (30, 1, 4), (40, 2, 5)]:
        for seed in range(1, 13):
            inst = S.barabasi_albert_instance(n, m, m0, seed)
            adj = A.adjacency(inst.n, inst.edges)
            means = [A.sir_many_runs(adj, beta, mu=0.3, n_runs=40, seed=seed, patient_zero=0, max_steps=60)["mean"] for beta in betas]
            for i in range(len(means) - 1):
                checked += 1
                if means[i] > means[i + 1] + 1.5:             # Toleranzband gegen Stichprobenrauschen
                    violations += 1
    assert checked >= 100
    assert violations / checked < 0.05                        # < 5% der Vergleiche duerfen (knapp) das Band verletzen


def test_mean_outbreak_size_non_decreasing_large_n_runs_tight_band():
    """Dieselbe Aussage mit deutlich mehr Laeufen je Punkt (n_runs=150) auf einer festen Instanz - das Toleranzband darf entsprechend enger sein."""
    inst = S.barabasi_albert_instance(60, 2, 4, 3)
    adj = A.adjacency(inst.n, inst.edges)
    betas = (0.02, 0.1, 0.3, 0.6, 0.95)
    means = [A.sir_many_runs(adj, beta, mu=0.3, n_runs=150, seed=11, patient_zero=0, max_steps=80)["mean"] for beta in betas]
    for i in range(len(means) - 1):
        assert means[i] <= means[i + 1] + 1.0


# --- Buchfuehrung/Sonderfaelle ------------------------------------------------------------------------------------------------------------------------------


def test_sir_determinism_same_seed_same_trace():
    inst = S.barabasi_albert_instance(25, 2, 4, 2)
    adj = A.adjacency(inst.n, inst.edges)
    a = A.sir_simulate(adj, beta=0.3, mu=0.2, seed=42, patient_zero=0, max_steps=40)
    b = A.sir_simulate(adj, beta=0.3, mu=0.2, seed=42, patient_zero=0, max_steps=40)
    assert a.trace == b.trace


def test_sir_invalid_patient_zero_raises():
    inst = S.barbell_instance(6)
    adj = A.adjacency(inst.n, inst.edges)
    with pytest.raises(ValueError):
        A.sir_simulate(adj, beta=0.1, mu=0.1, seed=1, patient_zero=999, max_steps=10)


def test_sir_n_le_2_and_isolated_patient_zero():
    for n, edges in [(1, []), (2, []), (2, [(0, 1, 1.0)])]:
        adj = A.adjacency(n, edges)
        result = A.sir_simulate(adj, beta=0.5, mu=0.5, seed=1, patient_zero=0, max_steps=20)
        assert result.final_s + result.final_i + result.final_r == n
        if n == 1 or not edges:
            assert result.outbreak_size == 1                 # kein Nachbar zum Anstecken


def test_sir_complete_graph_beta_one_mu_zero_infects_everyone_in_one_step():
    n = 8
    edges = [(i, j, 1.0) for i in range(n) for j in range(i + 1, n)]
    adj = A.adjacency(n, edges)
    result = A.sir_simulate(adj, beta=1.0, mu=0.0, seed=1, patient_zero=0, max_steps=5)
    assert result.outbreak_size == n
    assert result.trace[1] == (0, n, 0)                       # nach 1 Schritt schon alle infiziert (vollstaendiger Graph)
