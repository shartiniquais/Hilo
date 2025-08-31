from engine import Card, GameConfig, new_game, finish_round_and_score, COLOR_MAP


def test_scoring_trigger_unique_lowest_no_double():
    cfg = GameConfig(
        players=[("AI", "A0"), ("AI", "A1")],
        ai_depth=1,
        ai_mc_samples=1,
        ai_mc_seed=1,
        color_map=COLOR_MAP,
        score_limit=100,
    )
    state = new_game(cfg)
    # Player 0 will be trigger and has unique lowest sum 8
    g0 = state.players[0].grid
    g1 = state.players[1].grid
    g0.cells = [[None for _ in range(3)] for _ in range(3)]
    g0.visible = [[False for _ in range(3)] for _ in range(3)]
    g1.cells = [[None for _ in range(3)] for _ in range(3)]
    g1.visible = [[False for _ in range(3)] for _ in range(3)]
    g0.cells[0][0] = Card("R", 8)
    g0.visible[0][0] = True
    g1.cells[0][0] = Card("B", 9)
    g1.visible[0][0] = True

    state.end_trigger_idx = 0
    added = dict(finish_round_and_score(state))
    assert added["A0"] == 8  # no double
    assert added["A1"] == 9

