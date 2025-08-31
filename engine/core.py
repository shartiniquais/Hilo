from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Sequence, Dict, Any, Mapping, cast

from .types import Card, GridPos, PlayerKind
from .grid import Grid, apply_hilo_chain_pure, choose_best_compaction_pure
from .deck import DeckTracker
from .ai import CandidateEval, evaluate_best_place


def _append_log(state: "GameState", msg: str) -> None:
    if state.logs is None:
        state.logs = []
    state.logs.append(msg)


def _top_discard(state: "GameState") -> Optional[Card]:
    return state.discard[-1] if state.discard else None


def _put_on_discard(state: "GameState", cards: Sequence[Card]) -> None:
    # Smallest value on top -> append descending
    cards_sorted = sorted(cards, key=lambda c: c.v, reverse=True)
    for c in cards_sorted:
        state.discard.append(c)
        state.deck.see_card(c)


def _kw(obj: object) -> Mapping[str, Any]:
    # Helper to satisfy type-checker for **kwargs access
    assert isinstance(obj, dict)
    return cast(Mapping[str, Any], obj)


@dataclass
class Player:
    id: str           # "p0", "p1", ...
    name: str
    kind: PlayerKind  # Literal["H","AI"]
    grid: Grid
    total_points: int = 0


@dataclass
class GameConfig:
    players: List[Tuple[PlayerKind, str]]  # [(kind,name),... 2..4 active]
    ai_depth: Literal[1, 2]
    ai_mc_samples: int
    ai_mc_seed: int
    color_map: Dict[str, str]
    score_limit: int = 100


@dataclass
class ExplainInfo:
    topK: List[CandidateEval]
    pick_reason: str


@dataclass
class GameState:
    cfg: GameConfig
    players: List[Player]
    current_idx: int
    discard: List[Card]
    deck: DeckTracker
    end_trigger_idx: Optional[int] = None
    logs: List[str] = field(default_factory=list)
    explain: Optional[ExplainInfo] = None


def new_game(cfg: GameConfig) -> GameState:
    pls: List[Player] = []
    for i, (kind, name) in enumerate(cfg.players):
        pid = f"p{i}"
        pls.append(Player(id=pid, name=name, kind=kind, grid=Grid()))
    state = GameState(
        cfg=cfg,
        players=pls,
        current_idx=0,
        discard=[],
        deck=DeckTracker(),
        end_trigger_idx=None,
        logs=[],
        explain=None,
    )
    return state


def _player(state: GameState) -> Player:
    return state.players[state.current_idx]


def start_round(
    state: GameState,
    initial_reveals: Dict[str, List[Tuple[GridPos, Card]]],
    initial_discard: Card,
    starting_rule: Literal["lowest_two_sum", "highest_two_sum"] = "lowest_two_sum",
) -> None:
    state.discard = []
    state.deck.reset()
    for p in state.players:
        p.grid = Grid()
    # Apply initial reveals
    start_sums: List[Tuple[int, int]] = []  # (sum, idx)
    for idx, p in enumerate(state.players):
        picks = initial_reveals.get(p.id, [])
        ssum = 0
        for (r, c), card in picks:
            p.grid.reveal_known(r, c, card)
            state.deck.see_card(card)
            ssum += card.v
        start_sums.append((ssum, idx))
    # Determine starting player
    if starting_rule == "highest_two_sum":
        chosen_idx = max(start_sums, key=lambda t: t[0])[1]
    else:
        chosen_idx = min(start_sums, key=lambda t: t[0])[1]
    state.current_idx = chosen_idx

    # Set initial discard
    _put_on_discard(state, [initial_discard])
    _append_log(state, f"INIT_DISCARD: {_top_discard(state)}")


def is_round_over(state: GameState) -> bool:
    if state.end_trigger_idx is None:
        return False
    # Round ends when we are back to the trigger's turn
    return state.current_idx == state.end_trigger_idx


def _apply_hilo_chain(state: GameState, g: Grid) -> None:
    while True:
        lines = g.find_hilo_lines()
        if not lines:
            return
        line = lines[0]
        trio: List[Card] = []
        for (r, c) in line:
            card = g.cells[r][c]
            assert card is not None
            trio.append(card)
        typ = g.remove_line(line)
        _put_on_discard(state, trio)
        if typ == "diag":
            mode = choose_best_compaction_pure(g)
            _append_log(state, f"HILO_COMPACT: diag -> {mode}")
            g.compact_after_diag(mode)


def apply_human_action(
    state: GameState,
    *,
    action: Literal["take_discard", "draw_place", "draw_discard_flip"],
    target: Optional[GridPos] = None,
    drawn: Optional[Card] = None,
    replaced_hidden: Optional[Card] = None,
    flipped: Optional[GridPos] = None,
    flipped_card: Optional[Card] = None,
) -> None:
    p = _player(state)
    g = p.grid
    if action == "take_discard":
        top = _top_discard(state)
        assert top is not None, "Discard is empty"
        assert target is not None
        r, c = target
        replaced_opt = g.cells[r][c]
        assert replaced_opt is not None
        # Decide what to put on discard: known visible card or required hidden identity
        replaced_to_discard: Optional[Card]
        if g.visible[r][c]:
            replaced_to_discard = replaced_opt
        else:
            assert replaced_hidden is not None, "Must provide replaced hidden card"
            replaced_to_discard = replaced_hidden
        _append_log(state, f"HUMAN_ACTION: discard; PLACE: ({r},{c})")
        if not g.visible[r][c] and replaced_to_discard is not None:
            state.deck.see_card(replaced_to_discard)
            _append_log(state, f"HUMAN_REPLACED: {replaced_to_discard}")
        state.discard.pop()
        g.cells[r][c] = top
        g.visible[r][c] = True
        if replaced_to_discard is not None:
            _put_on_discard(state, [replaced_to_discard])
        _apply_hilo_chain(state, g)
    elif action == "draw_place":
        assert drawn is not None
        state.deck.see_card(drawn)
        assert target is not None
        r, c = target
        replaced_opt = g.cells[r][c]
        replaced: Optional[Card] = replaced_opt
        if replaced is None:
            # Fallback to first non-empty
            for rr in range(3):
                for cc in range(3):
                    if g.cells[rr][cc] is not None:
                        r, c = rr, cc
                        replaced = g.cells[r][c]
                        break
                if replaced is not None:
                    break
        assert replaced is not None
        # What goes to discard: if target was visible, the visible card; if hidden, use required provided identity
        if g.visible[r][c]:
            replaced_to_discard = replaced
        else:
            assert replaced_hidden is not None, "Must provide replaced hidden card"
            replaced_to_discard = replaced_hidden
        _append_log(state, f"HUMAN_ACTION: draw; PLACE: ({r},{c})")
        if not g.visible[r][c] and replaced_to_discard is not None:
            state.deck.see_card(replaced_to_discard)
            _append_log(state, f"HUMAN_REPLACED: {replaced_to_discard}")
        g.cells[r][c] = drawn
        g.visible[r][c] = True
        if replaced_to_discard is not None:
            _put_on_discard(state, [replaced_to_discard])
        _apply_hilo_chain(state, g)
    elif action == "draw_discard_flip":
        assert drawn is not None
        state.deck.see_card(drawn)
        assert flipped is not None and flipped_card is not None
        r, c = flipped
        _put_on_discard(state, [drawn])
        _append_log(state, f"HUMAN_DISCARDED: {drawn}")
        state.deck.see_card(flipped_card)
        g.reveal_known(r, c, flipped_card)
        _append_log(state, f"HUMAN_FLIPPED: ({r},{c}) -> {flipped_card}")
        _apply_hilo_chain(state, g)
    else:
        raise ValueError("Unknown action")


def ai_decide(state: GameState) -> Tuple[str, Dict, ExplainInfo]:
    p = _player(state)
    g = p.grid
    top = _top_discard(state)
    explain_topk: List[CandidateEval] = []
    pick_reason = ""
    # Evaluate take_discard if available
    take_discard_score: Optional[Tuple[float, GridPos]] = None
    take_discard_best: Optional[CandidateEval] = None
    if top is not None:
        td_score, td_best, td_cands = evaluate_best_place(
            g,
            top,
            state.deck,
            state.cfg.ai_depth,
            state.cfg.ai_mc_samples,
            state.cfg.ai_mc_seed,
            top_discard=top,
        )
        take_discard_score = td_score
        take_discard_best = td_best
        explain_topk = sorted(td_cands, key=lambda ce: ce.result_score)[:3]
        ev_after_td = td_best.baseline + td_best.ev_delta
        pick_reason += f"AI_EVAL: take_discard -> EV={ev_after_td:.3f}\n"

    # Heuristic: prefer discard if immediate improvement, else draw first
    draw_first = True
    if take_discard_score is not None:
        best_delta, _pos = take_discard_score
        if best_delta < 0.0:
            draw_first = False
    if draw_first:
        pick_reason += "AI_PICK: draw (expected benefit over discard)"
        info = ExplainInfo(topK=explain_topk, pick_reason=pick_reason)
        state.explain = info
        return ("draw", {}, info)
    else:
        assert take_discard_best is not None and take_discard_score is not None
        _, pos = take_discard_score
        r, c = pos
        if take_discard_best.kind == "hidden":
            pick_reason += (
                f"AI_PICK: take_discard -> PLACE=({r},{c}) hidden; p_improve={take_discard_best.p_improve:.3f}"
            )
        else:
            vk = take_discard_best.v_known if take_discard_best.v_known is not None else -999
            pick_reason += f"AI_PICK: take_discard -> PLACE=({r},{c}) known={vk}"
        info = ExplainInfo(topK=explain_topk, pick_reason=pick_reason)
        state.explain = info
        return ("take_discard", {"target": (r, c)}, info)


def ai_apply(state: GameState, action: str, **kwargs: object) -> None:
    p = _player(state)
    g = p.grid
    if action == "draw":
        _append_log(state, "AI_ACTION: draw")
        return
    if action == "take_discard":
        target_obj = _kw(kwargs).get("target")
        assert target_obj is not None
        target = cast(GridPos, target_obj)
        r, c = target
        top = _top_discard(state)
        assert top is not None
        replaced = g.cells[r][c]
        assert replaced is not None
        _append_log(state, f"AI_ACTION: discard; PLACE: ({r},{c})")
        if not g.visible[r][c]:
            # AI replaced hidden; must provide identity
            rh_obj = _kw(kwargs).get("replaced_hidden")
            assert rh_obj is not None, "AI replace hidden requires replaced_hidden"
            rh = cast(Card, rh_obj)
            replaced = rh
            state.deck.see_card(rh)
            _append_log(state, f"AI_REPLACED: {rh}")
        state.discard.pop()
        g.cells[r][c] = top
        g.visible[r][c] = True
        _put_on_discard(state, [replaced])
        _apply_hilo_chain(state, g)
        return
    if action == "draw_place":
        drawn = cast(Card, _kw(kwargs)["drawn"])        
        target = cast(GridPos, _kw(kwargs)["target"])   
        state.deck.see_card(drawn)
        r, c = target
        replaced = g.cells[r][c]
        if replaced is None:
            # Fallback to first non-empty
            for rr in range(3):
                for cc in range(3):
                    if g.cells[rr][cc] is not None:
                        r, c = rr, cc
                        replaced = g.cells[r][c]
                        break
                if replaced is not None:
                    break
        assert replaced is not None
        _append_log(state, f"AI_ACTION: draw; PLACE: ({r},{c})")
        replaced_card: Card = replaced
        if not g.visible[r][c]:
            rh_obj = _kw(kwargs).get("replaced_hidden")
            assert rh_obj is not None, "AI replace hidden requires replaced_hidden"
            rh = cast(Card, rh_obj)
            replaced_card = rh
            state.deck.see_card(rh)
            _append_log(state, f"AI_REPLACED: {rh}")
        g.cells[r][c] = drawn
        g.visible[r][c] = True
        _put_on_discard(state, [replaced_card])
        _apply_hilo_chain(state, g)
        return
    if action == "draw_discard_flip":
        d = _kw(kwargs)
        drawn = cast(Card, d["drawn"])  
        flipped = cast(GridPos, d["flipped"])  
        flipped_card = cast(Card, d["flipped_card"])  
        state.deck.see_card(drawn)
        _put_on_discard(state, [drawn])
        _append_log(state, f"AI_DISCARDED: {drawn}")
        r, c = flipped
        state.deck.see_card(flipped_card)
        g.reveal_known(r, c, flipped_card)
        _append_log(state, f"AI_FLIPPED: ({r},{c}) -> {flipped_card}")
        _apply_hilo_chain(state, g)
        return
    raise ValueError(f"Unknown AI action: {action}")


def maybe_trigger_end(state: GameState) -> None:
    if state.end_trigger_idx is not None:
        return
    p = _player(state)
    if p.grid.all_visible_or_empty():
        state.end_trigger_idx = state.current_idx
        _append_log(state, f"END_TRIGGER: {p.name}")


def finish_round_and_score(state: GameState) -> List[Tuple[str, int]]:
    # Assumes all remaining cards are already revealed by the CLI.
    totals: List[Tuple[int, int]] = []  # (sum, idx)
    for idx, p in enumerate(state.players):
        totals.append((p.grid.full_sum(), idx))
    # Compute doubling rule
    min_sum = min(s for s, _ in totals)
    min_count = sum(1 for s, _ in totals if s == min_sum)
    added: List[Tuple[str, int]] = []
    for s, idx in totals:
        p = state.players[idx]
        final = s
        if state.end_trigger_idx is not None and idx == state.end_trigger_idx:
            if s > min_sum or (s == min_sum and min_count >= 2):
                final = s * 2
        p.total_points += final
        added.append((p.name, final))
    return added


def is_game_over(state: GameState) -> bool:
    return any(p.total_points >= state.cfg.score_limit for p in state.players)
