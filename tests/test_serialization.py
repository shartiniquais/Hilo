from typing import Any, Dict, cast

from engine import (
    Card,
    GameConfig,
    new_game,
    apply_human_action,
    ai_decide,
    ai_apply,
    COLOR_MAP,
    to_json,
    from_json,
)


def test_round_trip_base():
    cfg = GameConfig(
        players=[("H", "You"), ("AI", "Bot")],
        ai_depth=1,
        ai_mc_samples=32,
        ai_mc_seed=42,
        color_map=COLOR_MAP,
        score_limit=100,
    )
    state0 = new_game(cfg)
    data = to_json(state0)
    state1 = from_json(data)
    assert to_json(state1) == data


def test_hilo_and_compaction_serialization():
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
    # Prepare two visible cards R3, R2 in row 0, and complete with R1
    g.reveal_known(0, 0, Card("R", 3))
    g.reveal_known(0, 1, Card("R", 2))
    apply_human_action(
        state,
        action="draw_place",
        target=(0, 2),
        drawn=Card("R", 1),
        replaced_hidden=Card("B", 4),
    )
    # Serialize then deserialize
    s = cast(Dict[str, Any], to_json(state))
    state2 = from_json(s)
    s2 = cast(Dict[str, Any], to_json(state2))
    # Check removed cells in row 0 are serialized as card=null and visible=false
    row0 = s2["players"][0]["grid"][0]
    for cell in row0:
        assert cell["card"] is None
        assert cell["visible"] is False
    # Discard top matches
    assert s2["discardTop"] == s["discardTop"]


def test_deck_reconstruction_sanity():
    cfg = GameConfig(
        players=[("H", "P0")],
        ai_depth=1,
        ai_mc_samples=1,
        ai_mc_seed=1,
        color_map=COLOR_MAP,
        score_limit=100,
    )
    state = new_game(cfg)
    g = state.players[0].grid
    # Reveal a few knowns and create a discard event to influence deck
    g.reveal_known(0, 0, Card("R", 5))
    g.reveal_known(1, 1, Card("B", 7))
    # Discard a drawn card to the pile
    apply_human_action(
        state,
        action="draw_discard_flip",
        drawn=Card("G", 3),
        flipped=(2, 2),
        flipped_card=Card("M", 9),
    )
    s = cast(Dict[str, Any], to_json(state))
    s2 = cast(Dict[str, Any], to_json(from_json(s)))
    assert s2["deckSummary"]["remainingTotal"] == s["deckSummary"]["remainingTotal"]
    assert s2["deckSummary"]["colorsDist"] == s["deckSummary"]["colorsDist"]


def test_explain_passthrough_topk():
    cfg = GameConfig(
        players=[("AI", "A0")],
        ai_depth=1,
        ai_mc_samples=32,
        ai_mc_seed=1337,
        color_map=COLOR_MAP,
        score_limit=100,
    )
    state = new_game(cfg)
    # Seed a simple visible card so evaluate has something stable
    p = state.players[0]
    p.grid.reveal_known(0, 0, Card("R", 0))
    # Provide a known top discard to consider
    state.discard = [Card("B", 1)]
    action, kwargs, _info = ai_decide(state)
    # Capture explain populated by ai_decide
    assert action in ("draw", "take_discard")
    s = cast(Dict[str, Any], to_json(state))
    assert "explain" in s
    topk = s["explain"]["topK"]
    assert isinstance(topk, list) and len(topk) >= 1
    keys = set(topk[0].keys())
    # Required fields
    for k in ("r", "c", "kind", "result", "delta", "p_improve", "ev_hidden", "hilo", "removed_sum"):
        assert k in keys


def test_engine_has_no_io_calls():
    import os
    bad = []
    for root, _dirs, files in os.walk("engine"):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
                if "print(" in txt or "input(" in txt:
                    bad.append(path)
    assert not bad, f"I/O found in engine modules: {bad}"
