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
    step,
    resolve,
    finish_round_and_score,
    is_game_over,
    COLOR_MAP,
    to_json,
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


def handle_pending(state: GameState) -> None:
    # Poll pending and ask user to resolve one by one
    while True:
        step(state)
        data = to_json(state)
        pend = data.get("pending", [])
        if not pend:
            break
        assert isinstance(pend, list)
        pa = pend[0]
        kid = pa["id"]
        kind = pa["kind"]
        payload = pa.get("payload", {})
        player_id = pa.get("playerId")
        p = next(pp for pp in state.players if pp.id == player_id)
        print()
        print_legend()
        print(f"Pending for {p.name} [{kind}]")
        # Show grid snapshot for that player
        print_grid(p.grid, f"{p.name}'s grid")
        if kind == "choose_action":
            allowed = payload.get("allowed", [])
            while True:
                choice = input(f"Choose action {allowed}: ").strip().lower()
                if choice in allowed:
                    break
            resolve(state, kid, {"choice": choice})
        elif kind == "choose_pos":
            r, c = ask_pos_any()
            resolve(state, kid, {"row": r, "col": c})
        elif kind in ("reveal_card", "replace_hidden_card", "ai_reveal_needed"):
            card = ask_card("Provide card (color+value):")
            resolve(state, kid, {"c": card.c, "v": card.v})
        elif kind == "choose_diag_compact":
            while True:
                s = input("Compaction 'vertical' or 'horizontal': ").strip().lower()
                if s in ("vertical", "horizontal"):
                    break
            resolve(state, kid, {"choice": s})
        else:
            print(f"Unknown pending kind: {kind}; skipping")
            break
        drain_logs(state)


def reveal_remaining_and_resolve(state: GameState) -> None:
    # The engine's step/resolve should handle reveals during play; this helper
    # just ensures all hidden cells are revealed before scoring (arbiter-driven).
    for p in state.players:
        g = p.grid
        for r in range(3):
            for c in range(3):
                if g.cells[r][c] is not None and not g.visible[r][c]:
                    print(f"Reveal remaining for {p.name} at ({r},{c})")
                    card = ask_card("Color+value:")
                    g.reveal_known(r, c, card)
                    state.deck.see_card(card)


def play_round(state: GameState) -> None:
    print("\n=== New Round ===")
    # Drive round using event system
    handle_pending(state)
    print(f"Initial discard top: {state.discard[-1] if state.discard else '(empty)'}")

    while True:
        handle_pending(state)
        # Step returns with no pending when round is over or needs no input
        if state.end_trigger_idx is not None and state.current_idx == state.end_trigger_idx:
            break
    # Reveal remaining and score
    reveal_remaining_and_resolve(state)
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
