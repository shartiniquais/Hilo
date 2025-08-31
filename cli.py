from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from engine import (
    Card,
    Grid,
    GridPos,
    Player,
    PlayerKind,
    GameConfig,
    GameState,
    new_game,
    start_round,
    is_round_over,
    apply_human_action,
    ai_decide,
    ai_apply,
    maybe_trigger_end,
    finish_round_and_score,
    is_game_over,
    COLOR_MAP,
    evaluate_best_place,
)


# Configuration migrated from the legacy file
PLAYERS: List[Tuple[PlayerKind, str]] = [
    ("H", "You"),
    ("AI", "Bot A"),
    ("AI", "Bot B"),
]

AI_DEPTH: int = 2
AI_MC_SAMPLES: int = 512
AI_MC_SEED: int = 1337
SCORE_LIMIT: int = 10


def print_legend() -> None:
    items = ", ".join(f"{k}={v}" for k, v in COLOR_MAP.items())
    print(f"Legend: {items}")


def parse_color_letter() -> str:
    while True:
        s = input("Color letter {R,B,G,Y,P,O,L,M}: ").strip().upper()
        if s in COLOR_MAP:
            return s
        print("Invalid color. Use letters R,B,G,Y,P,O,L,M.")


def parse_value() -> int:
    while True:
        s = input("Value (-1..11): ").strip()
        try:
            v = int(s)
        except Exception:
            print("Invalid number.")
            continue
        if -1 <= v <= 11:
            return v
        print("Out of range; must be -1..11.")


def ask_card(prompt: str = "Card") -> Card:
    print(prompt)
    col = parse_color_letter()
    val = parse_value()
    return Card(col, val)


def print_grid(g: Grid, title: str = "") -> None:
    if title:
        print(f"--- {title} ---")
    for r in range(3):
        row_parts: List[str] = []
        for c in range(3):
            card = g.cells[r][c]
            if card is None:
                row_parts.append("..")
            elif g.visible[r][c]:
                row_parts.append(f"{card.c}{card.v:>2}")
            else:
                row_parts.append(" ?")
        print(" ".join(f"{x:>3}" for x in row_parts))
    print()


def ask_pos_any() -> GridPos:
    while True:
        s = input("Target (row col) in 0..2 0..2: ").strip().split()
        if len(s) != 2:
            print("Please input two integers.")
            continue
        try:
            r, c = int(s[0]), int(s[1])
        except Exception:
            print("Invalid integers.")
            continue
        if 0 <= r <= 2 and 0 <= c <= 2:
            return (r, c)
        print("Out of bounds; use 0,1,2 for both.")


def ask_pos_hidden(g: Grid) -> GridPos:
    while True:
        (r, c) = ask_pos_any()
        if g.cells[r][c] is None:
            print("That cell is empty.")
            continue
        if g.visible[r][c]:
            print("That cell is already face-up; pick a hidden one.")
            continue
        return (r, c)


def first_hidden(g: Grid) -> GridPos:
    for r in range(3):
        for c in range(3):
            if g.cells[r][c] is not None and not g.visible[r][c]:
                return (r, c)
    for r in range(3):
        for c in range(3):
            if g.cells[r][c] is not None:
                return (r, c)
    return (0, 0)


def next_idx(state: GameState) -> int:
    n = len(state.players)
    return (state.current_idx + 1) % n


def drain_logs(state: GameState) -> None:
    for line in state.logs:
        print(line)
    state.logs.clear()


def human_turn(state: GameState) -> None:
    p = state.players[state.current_idx]
    g = p.grid
    print()
    print_legend()
    print(f"Turn: {p.name}")
    top = state.discard[-1] if state.discard else None
    print(f"Discard top: {top if top else '(empty)'}")
    print_grid(g, f"{p.name}'s grid")

    action: Optional[str] = None
    while action not in ("discard", "draw"):
        a = input("Action (take 'discard' or 'draw'): ").strip().lower()
        if a in ("discard", "draw"):
            action = a
    if action == "discard":
        if top is None:
            print("Discard is empty; you must draw.")
            action = "draw"
        else:
            r, c = ask_pos_any()
            replaced = g.cells[r][c]
            replaced_hidden: Optional[Card] = None
            if replaced is None:
                print("That cell is empty; pick a non-empty position.")
                return human_turn(state)
            if not g.visible[r][c]:
                replaced_hidden = ask_card("Replaced hidden card (color+value):")
            apply_human_action(state,
                               action="take_discard",
                               target=(r, c),
                               replaced_hidden=replaced_hidden)
            drain_logs(state)
            return

    # draw branch
    drawn = ask_card("Drawn card (color+value):")
    while True:
        sub = input("Place or discard? (p/d): ").strip().lower()
        if sub in ("p", "d"):
            break
    if sub == "p":
        r, c = ask_pos_any()
        replaced_hidden: Optional[Card] = None
        if not g.visible[r][c]:
            replaced_hidden = ask_card("Replaced hidden card (color+value):")
        apply_human_action(state,
                           action="draw_place",
                           target=(r, c),
                           drawn=drawn,
                           replaced_hidden=replaced_hidden)
    else:
        r, c = ask_pos_hidden(g)
        flip = ask_card("Revealed card (color+value):")
        apply_human_action(state,
                           action="draw_discard_flip",
                           drawn=drawn,
                           flipped=(r, c),
                           flipped_card=flip)
    drain_logs(state)


def ai_turn(state: GameState) -> None:
    p = state.players[state.current_idx]
    g = p.grid
    print()
    print_legend()
    print(f"Turn: {p.name} [AI]")
    top = state.discard[-1] if state.discard else None
    print(f"Discard top: {top if top else '(empty)'}")
    print_grid(g, f"{p.name}'s grid")

    action, kwargs, info = ai_decide(state)
    if info.pick_reason:
        print(info.pick_reason)
    if action == "take_discard":
        ai_apply(state, action, **kwargs)
        drain_logs(state)
        return
    # draw path
    print("AI_ACTION: draw")
    drawn = ask_card("AI needs the drawn card, please input color+value:")
    # Re-evaluate after draw
    (score, pos), best_cand, cand_list = evaluate_best_place(
        g,
        drawn,
        state.deck,
        state.cfg.ai_depth,
        state.cfg.ai_mc_samples,
        state.cfg.ai_mc_seed,
        top_discard=top,
    )
    r, c = pos
    # Compare draw_place vs draw_discard_flip
    can_flip_eval = g.count_hidden() > 0
    exp_draw = state.deck.expected_value_unseen()
    ev_after_draw_place = best_cand.baseline + best_cand.ev_delta
    ev_after_draw_discard = float(g.known_sum()) + (exp_draw if can_flip_eval else 0.0)
    place = (not can_flip_eval) or (ev_after_draw_place <= ev_after_draw_discard)
    if place:
        replaced_hidden: Optional[Card] = None
        if not g.visible[r][c]:
            replaced_hidden = ask_card("AI replaced a hidden card; please input it:")
            print(f"AI_REPLACED: {replaced_hidden}")
        ai_apply(state, "draw_place", drawn=drawn, target=(r, c), replaced_hidden=replaced_hidden)
    else:
        fr, fc = first_hidden(g)
        print(f"AI_ACTION: draw; DISCARD; FLIP: ({fr},{fc})")
        print(f"AI_DISCARDED: {drawn}")
        revealed = ask_card("Flip result for AI (color+value):")
        ai_apply(state, "draw_discard_flip", drawn=drawn, flipped=(fr, fc), flipped_card=revealed)
    drain_logs(state)


def reveal_remaining_and_resolve(state: GameState) -> None:
    # Ask reveals for any remaining hidden cards, then resolve chains
    for p in state.players:
        g = p.grid
        for r in range(3):
            for c in range(3):
                if g.cells[r][c] is not None and not g.visible[r][c]:
                    print(f"Reveal remaining for {p.name} at ({r},{c})")
                    card = ask_card("Color+value:")
                    state.deck.see_card(card)
                    g.reveal_known(r, c, card)
        # Resolve HILO chains deterministically (best compaction automatically)
        # Using draw_discard_flip with a dummy path would be awkward; the engine resolves on placement
        # so here we simply re-run chain resolution by simulating internal helper via a no-op replace.
        # This is implicitly done as part of scoring in the legacy, but we call it explicitly by placing nothing.
        # No additional logs printed here.
        from engine.core import _apply_hilo_chain as _internal_apply
        _internal_apply(state, g)


def play_round(state: GameState) -> None:
    # Collect initial reveals
    initial_reveals: Dict[str, List[Tuple[GridPos, Card]]] = {}
    print("\n=== New Round ===")
    print("Enter initial two reveals per active player.")
    for p in state.players:
        g = p.grid
        print()
        print_legend()
        print(f"Initial reveals for {p.name} (two positions)")
        picks: List[GridPos] = []
        while len(picks) < 2:
            r, c = ask_pos_any()
            if (r, c) in picks:
                print("Duplicate position; pick another.")
                continue
            card = ask_card(f"Card at ({r},{c}):")
            picks.append((r, c))
            initial_reveals.setdefault(p.id, []).append(((r, c), card))
    # Initial discard
    print()
    print_legend()
    init_top = ask_card("Initial discard (color+value):")
    start_round(state, initial_reveals, init_top, starting_rule="highest_two_sum")
    print(f"Initial discard top: {state.discard[-1] if state.discard else '(empty)'}")

    while True:
        if is_round_over(state):
            break
        p = state.players[state.current_idx]
        if p.kind == "H":
            human_turn(state)
        else:
            ai_turn(state)
        maybe_trigger_end(state)
        top = state.discard[-1] if state.discard else None
        print(f"Discard top now: {top if top else '(empty)'}")
        state.current_idx = next_idx(state)

    # Reveal and resolve remaining
    reveal_remaining_and_resolve(state)

    # Show final grids and score
    for p in state.players:
        print(f"\nFinal grid for {p.name}:")
        print_grid(p.grid)
    added = finish_round_and_score(state)
    for name, pts in added:
        print(f"{name}: {pts} points")
    print("\nCumulative:")
    for p in state.players:
        print(f"  {p.name}: {p.total_points} pts")


def main() -> None:
    print("HILO — Console Arbiter (engine + adapter)")
    print_legend()
    cfg = GameConfig(
        players=PLAYERS,
        ai_depth=2,
        ai_mc_samples=AI_MC_SAMPLES,
        ai_mc_seed=AI_MC_SEED,
        color_map=COLOR_MAP,
        score_limit=SCORE_LIMIT,
    )
    state = new_game(cfg)
    while True:
        play_round(state)
        if is_game_over(state):
            break
    print("\n=== Game Over ===")
    best_total = min(p.total_points for p in state.players)
    winners = [p.name for p in state.players if p.total_points == best_total]
    for p in state.players:
        print(f"{p.name}: {p.total_points} pts")
    if len(winners) == 1:
        print(f"Winner: {winners[0]}")
    else:
        print("Winners (tie): " + ", ".join(winners))


if __name__ == "__main__":
    main()

