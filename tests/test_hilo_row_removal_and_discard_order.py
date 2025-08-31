from engine import Card, GameConfig, new_game, apply_human_action, COLOR_MAP


def test_hilo_row_removal_and_discard_order():
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
    # Place R1 at (0,2) via a draw action
    # Provide the identity of the hidden card being replaced to keep deck tracking realistic
    apply_human_action(state, action="draw_place", target=(0, 2), drawn=Card("R", 1), replaced_hidden=Card("B", 4))
    # Three cards should be added to discard with smallest on top: 3,2,1 appended -> top is 1
    assert len(state.discard) >= 3
    top = state.discard[-1]
    assert top.c == "R" and top.v == 1
    assert state.discard[-2].v == 2
    assert state.discard[-3].v == 3
