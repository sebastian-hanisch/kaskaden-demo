"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt für jede Instanzart und jede Dynamik, bedingte Regler, Permalink-Grenzen, m<=m0-Kopplung, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kas_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    state.setdefault("kas_step", step)
    state.setdefault("side_slider", 6)
    state.setdefault("nba_slider", 30)
    state.setdefault("kbarbell_slider", 5)
    state.setdefault("nruns_slider", 20)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()


def test_default_run_shows_the_summary():
    at = _run()
    _ok(at)
    assert {"Knoten", "Kanten"} <= {m.label for m in at.metric}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run(kind_select="city")
    _click(at, f"preset_{name}")
    _ok(at)
    p, ss = C.PRESETS[name], at.session_state
    assert ss["kind_select"] == p["kind"] and ss["kas_step"] == p["step"]


@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["kaskade", "ausbreitung"])
@pytest.mark.parametrize("kind", ["city", "ba", "barbell"])
def test_every_step_runs_for_every_kind_and_mode(step, mode, kind):
    at = _run(step=step, kind_select=kind, mode_select=mode)
    _ok(at)
    assert at.session_state["kas_step"] == step
    assert at.session_state["mode_select"] == mode


def _chart_count(at):
    return len(at.get("plotly_chart"))


def test_step2_only_shows_curve_in_kaskade_mode():
    at_kaskade = _run(step=2, mode_select="kaskade")
    _ok(at_kaskade)
    assert _chart_count(at_kaskade) >= 1
    at_ausbreitung = _run(step=2, mode_select="ausbreitung")
    _ok(at_ausbreitung)
    assert _chart_count(at_ausbreitung) == 0
    assert any("gilt nur im Kaskade-Modus" in c.value for c in at_ausbreitung.caption)


def test_step3_only_shows_curve_in_ausbreitung_mode():
    at_ausbreitung = _run(step=3, mode_select="ausbreitung")
    _ok(at_ausbreitung)
    assert _chart_count(at_ausbreitung) >= 1
    at_kaskade = _run(step=3, mode_select="kaskade")
    _ok(at_kaskade)
    assert _chart_count(at_kaskade) == 0
    assert any("gilt nur im Ausbreitung-Modus" in c.value for c in at_kaskade.caption)


def test_step4_shows_network_comparison_table_and_two_bar_charts():
    at = _run(step=4)
    _ok(at)
    assert len(at.dataframe) >= 1
    assert _chart_count(at) >= 2


def test_alpha_slider_only_shown_in_kaskade_mode():
    at_kaskade = _run(mode_select="kaskade")
    assert any(w.key == "alpha_widget" for w in at_kaskade.slider)
    at_ausbreitung = _run(mode_select="ausbreitung")
    assert not any(w.key == "alpha_widget" for w in at_ausbreitung.slider)


def test_beta_mu_nruns_sliders_only_shown_in_ausbreitung_mode():
    at_ausbreitung = _run(mode_select="ausbreitung")
    keys = {w.key for w in at_ausbreitung.slider}
    assert {"beta_widget", "mu_widget", "nruns_widget"} <= keys
    at_kaskade = _run(mode_select="kaskade")
    keys2 = {w.key for w in at_kaskade.slider}
    assert not ({"beta_widget", "mu_widget", "nruns_widget"} & keys2)


def test_blocked_slider_only_shown_for_grid_nettype():
    at = _run(kind_select="city", nettype_select="grid")
    assert any(w.key == "blocked_widget" for w in at.select_slider)
    at2 = _run(kind_select="city", nettype_select="random")
    assert not any(w.key == "blocked_widget" for w in at2.select_slider)


def test_sidebar_shows_the_controls_that_belong_to_the_instance():
    city = _run(kind_select="city")
    assert any(w.key == "side_widget" for w in city.slider) and not any(w.key == "nba_widget" for w in city.slider)
    ba = _run(kind_select="ba")
    assert any(w.key == "nba_widget" for w in ba.slider) and not any(w.key == "side_widget" for w in ba.slider)
    barbell = _run(kind_select="barbell")
    assert any(w.key == "kbarbell_widget" for w in barbell.slider) and not any(w.key == "nba_widget" for w in barbell.slider)


def test_ba_m_slider_never_exceeds_m0():
    at = _run(kind_select="ba", m0ba_slider=3, mba_slider=10)          # m gespeichert > neues m0 -> muss geklemmt werden
    _ok(at)
    assert at.session_state["mba_slider"] <= 3


def test_dice_button_changes_the_seed_and_the_visible_widget():
    at = _run(kind_select="city")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old and at.session_state["seed_widget"] == at.session_state["seed_input"]


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(kind="nope", mode="up", side="9999", blocked="0.33", nettype="sideways", nba="99999", kbarbell="9999", initial="up", alpha="9999", beta="9999", mu="9999", nruns="99999",
                      seed="-4", step="9").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert ss["kind_select"] == "city"
    assert ss["mode_select"] == "kaskade"
    assert ss["side_slider"] == C.SIDE_MAX
    assert ss["blocked_select"] == C.DEFAULT_BLOCKED
    assert ss["nettype_select"] == "grid"
    assert ss["nba_slider"] == C.N_BA_MAX
    assert ss["kbarbell_slider"] == C.BARBELL_K_MAX
    assert ss["initial_strategy_select"] == "highest_load"
    assert ss["alpha_slider"] == C.ALPHA_MAX
    assert ss["beta_slider"] == C.BETA_MAX
    assert ss["mu_slider"] == C.MU_MAX
    assert ss["nruns_slider"] == C.N_RUNS_MAX
    assert ss["seed_input"] == 0
    assert ss["kas_step"] == 1


def test_permalink_accepts_valid_values_and_writes_them_back():
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in dict(kind="ba", mode="ausbreitung", nba="60", m0ba="5", mba="2", initial="random", beta="0.2", mu="0.3", seed="7", step="3").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["mode_select"], ss["nba_slider"], ss["m0ba_slider"], ss["initial_strategy_select"], ss["seed_input"], ss["kas_step"]) == \
        ("ba", "ausbreitung", 60, 5, "random", 7, 3)
    assert at.query_params["seed"] == ["7"] and at.query_params["step"] == ["3"]
    assert ss["nba_widget"] == 60 and ss["seed_widget"] == 7


def test_switching_kind_back_and_forth_keeps_the_stored_values():
    at = _run(kind_select="city", side_slider=14, blocked_select=0.4, seed_input=11)
    at.session_state["kind_select"] = "ba"
    at.run()
    _ok(at)
    at.session_state["kind_select"] = "city"
    at.run()
    _ok(at)
    assert at.session_state["side_widget"] == 14 and at.session_state["blocked_widget"] == 0.4 and at.session_state["seed_widget"] == 11


@pytest.mark.parametrize("kw", [dict(kind_select="city", side_slider=C.SIDE_MIN), dict(kind_select="city", side_slider=C.SIDE_MAX), dict(kind_select="ba", nba_slider=C.N_BA_MIN),
                                 dict(kind_select="barbell", kbarbell_slider=C.BARBELL_K_MIN), dict(kind_select="barbell", kbarbell_slider=C.BARBELL_K_MAX)])
def test_extreme_settings_run_on_every_step_and_mode(kw):
    for step in (1, 2, 3, 4):
        for mode in ("kaskade", "ausbreitung"):
            _ok(_run(step=step, mode_select=mode, **kw))


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Motter" in m.value and "Pastor-Satorras" in m.value for e in at.expander for m in e.markdown)
