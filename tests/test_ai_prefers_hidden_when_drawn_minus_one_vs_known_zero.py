from engine import Card, GameConfig, new_game, evaluate_best_place, COLOR_MAP


def test_ai_prefers_hidden_when_drawn_minus_one_vs_known_zero():
    cfg = GameConfig(
        players=[("AI", "Bot")],
        ai_depth=1,  # no MC to keep it deterministic
        ai_mc_samples=128,
        ai_mc_seed=1337,
        color_map=COLOR_MAP,
        score_limit=100,
    )
    state = new_game(cfg)
    g = state.players[0].grid
    # Make one known zero and leave others hidden
    g.reveal_known(0, 0, Card("R", 0))
    # Drawn card is -1
    drawn = Card("B", -1)
    (score, pos), best_cand, cands = evaluate_best_place(
        g,
        drawn,
        state.deck,
        state.cfg.ai_depth,
        state.cfg.ai_mc_samples,
        state.cfg.ai_mc_seed,
        top_discard=None,
    )
    # Should prefer a hidden cell rather than the known 0
    assert best_cand.kind == "hidden"

