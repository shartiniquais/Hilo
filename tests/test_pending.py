from typing import Any, Dict, List, cast

from engine import (
    Card,
    GameConfig,
    GameState,
    COLOR_MAP,
    new_game,
    step,
    resolve,
    to_json,
)


def _drain_initial_setup(state: GameState) -> None:
    # Resolve two initial reveals per player and initial discard
    while True:
        step(state)
        data = to_json(state)
        pending = cast(List[Dict[str, Any]], data.get("pending", []))
        if not pending:
            break
        pa = pending[0]
        kind = pa["kind"]
        kid = pa["id"]
        payload = pa.get("payload", {})
        if kind == "reveal_card" and payload.get("context") == "initial_reveal":
            resolve(state, kid, {"c": "R", "v": 0})
        elif kind == "reveal_card" and payload.get("context") == "initial_discard":
            resolve(state, kid, {"c": "B", "v": 1})
        else:
            break


def test_human_initial_reveals_flow_and_deck_updates():
    cfg = GameConfig(
        players=[("H", "You"), ("H", "Friend")],
        ai_depth=1,
        ai_mc_samples=16,
        ai_mc_seed=1,
        color_map=COLOR_MAP,
        score_limit=100,
    )
    state = new_game(cfg)
    # First step should request a reveal
    step(state)
    s = cast(Dict[str, Any], to_json(state))
    pending0 = cast(List[Dict[str, Any]], s["pending"])
    assert pending0, "Expected a pending action"
    assert pending0[0]["kind"] == "reveal_card"
    # Resolve all initial setup
    _drain_initial_setup(state)
    # After setup, should be in playing phase with a choose_action pending for the starting player
    step(state)
    ss = cast(Dict[str, Any], to_json(state))
    pending1 = cast(List[Dict[str, Any]], ss["pending"]) if ss.get("pending") is not None else []
    assert pending1, "Expected pending choose_action for human turn"
    assert pending1[0]["kind"] in ("choose_action", "ai_reveal_needed", "reveal_card")


def test_ai_hidden_replacement_pending_and_resolution():
    cfg = GameConfig(
        players=[("AI", "Bot")],
        ai_depth=1,
        ai_mc_samples=64,
        ai_mc_seed=1337,
        color_map=COLOR_MAP,
        score_limit=100,
    )
    state = new_game(cfg)
    # Complete setup quickly
    _drain_initial_setup(state)
    # Drive until AI asks for a draw
    step(state)
    s1 = cast(Dict[str, Any], to_json(state))
    assert s1["pending"], "Expected pending for AI"
    pa = cast(List[Dict[str, Any]], s1["pending"])[0]
    assert pa["kind"] == "ai_reveal_needed"
    # Provide drawn card as very low value to encourage placement
    resolve(state, pa["id"], {"c": "G", "v": -1})
    # Next, expect replaced_hidden reveal if target is hidden
    step(state)
    s2 = cast(Dict[str, Any], to_json(state))
    assert s2["pending"], "Expected pending after AI draw"
    pa2 = cast(List[Dict[str, Any]], s2["pending"])[0]
    assert pa2["kind"] == "ai_reveal_needed"
    # Supply identity of replaced hidden
    resolve(state, pa2["id"], {"c": "R", "v": 5})
    # Game continues without crashing; logs should contain AI_ACTION lines
    step(state)
    logs = state.logs
    assert any("AI_ACTION" in ln for ln in logs)


def test_diag_compaction_asks_choice_and_applies():
    cfg = GameConfig(
        players=[("H", "P0")],
        ai_depth=1,
        ai_mc_samples=1,
        ai_mc_seed=1,
        color_map=COLOR_MAP,
        score_limit=100,
    )
    state = new_game(cfg)
    p = state.players[0]
    g = p.grid
    # Prepare two diagonal visible cards and complete with third
    g.reveal_known(0, 0, Card("R", 3))
    g.reveal_known(1, 1, Card("R", 2))
    # Place R1 at (2,2) via draw_place to trigger diag removal
    from engine import apply_human_action
    apply_human_action(state, action="draw_place", target=(2, 2), drawn=Card("R", 1), replaced_hidden=Card("B", 4))
    # Should pause for compaction choice
    s = cast(Dict[str, Any], to_json(state))
    assert s["pending"], "Expected choose_diag_compact pending"
    pa = cast(List[Dict[str, Any]], s["pending"])[0]
    assert pa["kind"] == "choose_diag_compact"
    # Choose vertical compaction
    resolve(state, pa["id"], {"choice": "vertical"})
    # After compaction, bottom row should be empty
    bottom_none = all(g.cells[2][c] is None for c in range(3))
    assert bottom_none


def test_logs_accumulate_after_step_resolve_cycles():
    cfg = GameConfig(
        players=[("AI", "Bot")],
        ai_depth=1,
        ai_mc_samples=32,
        ai_mc_seed=1337,
        color_map=COLOR_MAP,
        score_limit=100,
    )
    state = new_game(cfg)
    _drain_initial_setup(state)
    # Drive a couple of AI actions
    step(state)
    s1 = cast(Dict[str, Any], to_json(state))
    if s1["pending"]:
        resolve(state, cast(List[Dict[str, Any]], s1["pending"])[0]["id"], {"c": "B", "v": 0})
    step(state)
    s2 = cast(Dict[str, Any], to_json(state))
    if s2["pending"] and cast(List[Dict[str, Any]], s2["pending"])[0]["kind"] == "ai_reveal_needed":
        resolve(state, cast(List[Dict[str, Any]], s2["pending"])[0]["id"], {"c": "R", "v": 4})
    # Logs should contain AI_ACTION lines
    assert any(ln.startswith("AI_ACTION") for ln in state.logs)
