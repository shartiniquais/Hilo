from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Sequence, Dict, Any, Mapping, cast

from .types import Card, GridPos, PlayerKind, COLOR_MAP, PendingAction, PendingKind
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
    # Event system
    pending: List[PendingAction] = field(default_factory=list)
    _next_action_seq: int = 1
    _phase: Literal["setup_reveals", "setup_discard", "playing"] = "setup_reveals"
    _setup_reveals_done: Dict[str, int] = field(default_factory=dict)
    _human_ctx: Dict[str, Any] = field(default_factory=dict)
    _ai_ctx: Dict[str, Any] = field(default_factory=dict)


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
        pending=[],
        _next_action_seq=1,
        _phase="setup_reveals",
        _setup_reveals_done={},
        _human_ctx={},
        _ai_ctx={},
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
    # Eventful: resolve chains; pause for diagonal compaction decision
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
            # Request compaction decision from arbiter
            _push_pending(state, kind="choose_diag_compact", player_id=_player(state).id, payload={"options": ["vertical", "horizontal"]})
            return


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


# --- JSON serialization (pure, no I/O) ---

def _card_to_obj(card: Optional[Card]) -> Optional[Dict[str, object]]:
    if card is None:
        return None
    return {"c": card.c, "v": int(card.v)}


def _obj_to_card(obj: object) -> Optional[Card]:
    if obj is None:
        return None
    assert isinstance(obj, dict)
    c = obj.get("c")
    v = obj.get("v")
    assert isinstance(c, str) and len(c) >= 1, "Invalid card color"
    assert isinstance(v, int), "Invalid card value"
    # Validate against known colors/values
    assert c in COLOR_MAP, f"Unknown color code: {c}"
    assert -1 <= v <= 11, f"Invalid value: {v}"
    return Card(c, int(v))


def to_json(state: GameState) -> Dict[str, object]:
    # Config for UI
    cfg_obj: Dict[str, object] = {
        "aiDepth": int(state.cfg.ai_depth),
        "scoreLimit": int(state.cfg.score_limit),
        "colorMap": dict(state.cfg.color_map),
    }

    # Players ordered for table order
    players_obj: List[Dict[str, object]] = []
    for p in state.players:
        grid_rows: List[List[Dict[str, object]]] = []
        for r in range(3):
            row: List[Dict[str, object]] = []
            for c in range(3):
                vis = bool(p.grid.visible[r][c])
                cell_card = p.grid.cells[r][c]
                # Only expose card identity when visible; otherwise null
                row.append({
                    "card": _card_to_obj(cell_card) if vis else None,
                    "visible": vis,
                })
            grid_rows.append(row)
        players_obj.append({
            "id": p.id,
            "name": p.name,
            "kind": p.kind,
            "totalPoints": int(p.total_points),
            "grid": grid_rows,
        })

    # Discard top card
    top = _top_discard(state)
    discard_top_obj = _card_to_obj(top)

    # Deck summary (UI only) — recompute from visible information for stability
    # across round-trips and direct state mutations in tests.
    dt = DeckTracker()
    dt.reset()
    for p in state.players:
        for r in range(3):
            for c in range(3):
                if p.grid.visible[r][c]:
                    card = p.grid.cells[r][c]
                    if isinstance(card, Card):
                        dt.see_card(card)
    if top is not None:
        dt.see_card(top)
    # Best-effort: mirror log parsing used in from_json
    for line in state.logs:
        if not isinstance(line, str):
            continue
        for prefix in (
            "AI_REPLACED:",
            "HUMAN_REPLACED:",
            "AI_FLIPPED:",
            "HUMAN_FLIPPED:",
            "AI_DISCARDED:",
            "HUMAN_DISCARDED:",
            "INIT_DISCARD:",
        ):
            if prefix in line:
                try:
                    token = line.split(prefix, 1)[1].strip()
                    if "->" in token:
                        token = token.split("->", 1)[1].strip()
                    token = token.split()[0]
                    if len(token) >= 2:
                        cc = token[0]
                        vv = int(token[1:])
                        if cc in COLOR_MAP and -1 <= vv <= 11:
                            dt.see_card(Card(cc, vv))
                except Exception:
                    pass

    values_dist = dt.remaining_values_distribution()
    colors_dist = dt.remaining_colors_distribution()
    values_dist_str: Dict[str, int] = {str(k): int(v) for k, v in values_dist.items()}
    deck_summary = {
        "remainingTotal": int(dt.remaining_total()),
        "valuesDist": values_dist_str,
        "colorsDist": {k: int(v) for k, v in colors_dist.items()},
    }

    # Optional explain info
    explain_obj: Optional[Dict[str, object]] = None
    if state.explain is not None:
        topk_list: List[Dict[str, object]] = []
        for ce in state.explain.topK:
            topk_list.append({
                "r": int(ce.r),
                "c": int(ce.c),
                "kind": ce.kind,
                "result": float(ce.result_score),
                "delta": float(ce.delta_score),
                "p_improve": float(ce.p_improve),
                "ev_hidden": None if ce.ev_delta_hidden is None else float(ce.ev_delta_hidden),
                "hilo": bool(ce.immediate_hilo),
                "removed_sum": int(ce.removed_sum),
            })
        pick_obj: Dict[str, object] = {"reason": state.explain.pick_reason}
        explain_obj = {"topK": topk_list, "pick": pick_obj}

    # Pending actions
    pendings: List[Dict[str, object]] = []
    for pa in state.pending:
        pendings.append({
            "id": pa.id,
            "kind": pa.kind,
            "playerId": pa.playerId,
            "payload": pa.payload,
        })

    data: Dict[str, object] = {
        "schemaVersion": 1,
        "config": cfg_obj,
        "players": players_obj,
        "currentPlayerId": state.players[state.current_idx].id,
        "endTriggerPlayerId": None if state.end_trigger_idx is None else state.players[state.end_trigger_idx].id,
        "discardTop": discard_top_obj,
        "deckSummary": deck_summary,
        "logs": list(state.logs),
        "pending": pendings,
    }
    if explain_obj is not None:
        data["explain"] = explain_obj
    return data


def from_json(data: Dict[str, object]) -> GameState:
    # Basic validation
    assert isinstance(data, dict), "Data must be a dict"
    schema_version = data.get("schemaVersion")
    assert schema_version == 1, "Unsupported schemaVersion"

    # Config
    cfgd = data.get("config")
    assert isinstance(cfgd, dict), "Missing config"
    ai_depth = cfgd.get("aiDepth")
    score_limit = cfgd.get("scoreLimit")
    color_map_obj = cfgd.get("colorMap")
    assert ai_depth in (1, 2), "aiDepth must be 1 or 2"
    assert isinstance(score_limit, int), "scoreLimit must be int"
    assert isinstance(color_map_obj, dict), "colorMap must be a dict"
    # Validate color map contains Lime 'L'
    assert color_map_obj.get("L") == COLOR_MAP["L"], "Color map must include Lime (L)"

    # Players setup
    p_list = data.get("players")
    assert isinstance(p_list, list) and len(p_list) >= 1, "players list required"
    pl_cfg: List[Tuple[PlayerKind, str]] = []
    for pobj in p_list:
        assert isinstance(pobj, dict)
        kind = pobj.get("kind")
        name = pobj.get("name")
        assert kind in ("H", "AI"), "Invalid player kind"
        assert isinstance(name, str), "Invalid player name"
        pl_cfg.append((cast(PlayerKind, kind), name))

    # Fill non-serialized AI params with deterministic defaults
    cfg = GameConfig(
        players=pl_cfg,
        ai_depth=cast(Literal[1, 2], ai_depth),
        ai_mc_samples=128,
        ai_mc_seed=1337,
        color_map=cast(Dict[str, str], color_map_obj),
        score_limit=int(score_limit),
    )
    state = new_game(cfg)

    # Map back player details and grids
    players_by_id: Dict[str, int] = {p.id: idx for idx, p in enumerate(state.players)}
    for idx, pobj in enumerate(p_list):
        assert isinstance(pobj, dict)
        pid = pobj.get("id")
        name = pobj.get("name")
        kind = pobj.get("kind")
        total_points = pobj.get("totalPoints", 0)
        grid = pobj.get("grid")
        assert isinstance(pid, str) and pid in players_by_id, "Invalid player id"
        p = state.players[idx]
        # Enforce id/name/kind/points
        p.id = pid
        p.name = name  # type: ignore[assignment]
        p.kind = cast(PlayerKind, kind)  # type: ignore[assignment]
        p.total_points = int(total_points)
        # Grid
        assert isinstance(grid, list) and len(grid) == 3, "Grid must be 3 rows"
        for r in range(3):
            row = grid[r]
            assert isinstance(row, list) and len(row) == 3, "Grid row must have 3 cols"
            for c in range(3):
                cell = row[c]
                assert isinstance(cell, dict), "Invalid cell"
                vis = bool(cell.get("visible", False))
                card_obj = cell.get("card")
                if vis:
                    card = _obj_to_card(card_obj)
                    assert card is not None, "Visible cell must have card"
                    p.grid.cells[r][c] = card
                    p.grid.visible[r][c] = True
                else:
                    # Hidden or removed; we cannot distinguish without extra info.
                    # Represent as unknown hidden placeholder to preserve playability.
                    p.grid.cells[r][c] = Card("?", 0)
                    p.grid.visible[r][c] = False

    # Current player and optional end trigger
    cpid = data.get("currentPlayerId")
    assert isinstance(cpid, str), "currentPlayerId required"
    try:
        state.current_idx = next(i for i, p in enumerate(state.players) if p.id == cpid)
    except StopIteration:
        raise AssertionError("currentPlayerId not found in players")
    etpid = data.get("endTriggerPlayerId")
    if etpid is None:
        state.end_trigger_idx = None
    else:
        assert isinstance(etpid, str)
        try:
            state.end_trigger_idx = next(i for i, p in enumerate(state.players) if p.id == etpid)
        except StopIteration:
            state.end_trigger_idx = None

    # Discard top
    top_obj = data.get("discardTop")
    top_card = _obj_to_card(top_obj) if top_obj is not None else None
    state.discard = []
    if top_card is not None:
        state.discard.append(top_card)

    # Logs
    logs_obj = data.get("logs", [])
    assert isinstance(logs_obj, list)
    state.logs = [str(x) for x in logs_obj]

    # Explain (optional, best-effort reconstruction)
    exp = data.get("explain")
    if isinstance(exp, dict):
        topk_raw = exp.get("topK", [])
        pick = exp.get("pick", {})
        topk_list: List[CandidateEval] = []
        if isinstance(topk_raw, list):
            for itm in topk_raw:
                if not isinstance(itm, dict):
                    continue
                r = int(itm.get("r", 0))
                c = int(itm.get("c", 0))
                kind = itm.get("kind", "hidden")
                result = float(itm.get("result", 0.0))
                delta = float(itm.get("delta", 0.0))
                p_improve = float(itm.get("p_improve", 0.0))
                ev_hidden = itm.get("ev_hidden")
                ev_hidden_f = None if ev_hidden is None else float(ev_hidden)
                hilo = bool(itm.get("hilo", False))
                removed_sum = int(itm.get("removed_sum", 0))
                # Fill missing dataclass fields with neutral values
                ce = CandidateEval(
                    r=r,
                    c=c,
                    kind=cast(Literal["visible", "hidden"], kind),
                    result_score=result,
                    delta_score=delta,
                    p_improve=p_improve,
                    ev_delta_hidden=ev_hidden_f,
                    immediate_hilo=hilo,
                    removed_sum=removed_sum,
                    v_known=None,
                    baseline=0.0,
                    ev_delta=delta,
                )
                topk_list.append(ce)
        reason = ""
        if isinstance(pick, dict):
            reason = str(pick.get("reason", ""))
        state.explain = ExplainInfo(topK=topk_list, pick_reason=reason)

    # Recompute deck from visible information and known cards
    state.deck.reset()
    for p in state.players:
        for r in range(3):
            for c in range(3):
                if p.grid.visible[r][c]:
                    card = p.grid.cells[r][c]
                    assert isinstance(card, Card)
                    state.deck.see_card(card)
    if top_card is not None:
        state.deck.see_card(top_card)

    # Optionally parse logs to mark other known cards (replaced/flip/discard/init)
    for line in state.logs:
        if not isinstance(line, str):
            continue
        # Find substrings that look like single-card tokens like "R3", "L-1" etc.
        # We use simple patterns around known prefixes to avoid false positives.
        for prefix in ("AI_REPLACED:", "HUMAN_REPLACED:", "AI_FLIPPED:", "HUMAN_FLIPPED:", "AI_DISCARDED:", "HUMAN_DISCARDED:", "INIT_DISCARD:"):
            if prefix in line:
                try:
                    token = line.split(prefix, 1)[1].strip()
                    # For flipped lines, format is "(r,c) -> Cx"; take last token
                    if "->" in token:
                        token = token.split("->", 1)[1].strip()
                    token = token.split()[0]
                    # Color is first char; value is the rest, may be negative
                    if len(token) >= 2:
                        cc = token[0]
                        vv = int(token[1:])
                        if cc in COLOR_MAP and -1 <= vv <= 11:
                            state.deck.see_card(Card(cc, vv))
                except Exception:
                    # Best-effort; ignore malformed lines
                    pass
    
    # Pending actions (deserialize best-effort)
    p_list = data.get("pending", [])
    state.pending = []
    if isinstance(p_list, list):
        for itm in p_list:
            if not isinstance(itm, dict):
                continue
            kid = str(itm.get("id", ""))
            kk = str(itm.get("kind", ""))
            pid = str(itm.get("playerId", ""))
            payload = itm.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            try:
                state.pending.append(PendingAction(kind=cast(PendingKind, kk), playerId=pid, payload=payload, id=kid))
            except Exception:
                # ignore malformed entries
                pass

    return state


# --- Event system: step + resolve ---

def _next_id(state: GameState) -> str:
    i = state._next_action_seq
    state._next_action_seq += 1
    return f"a{i}"


def _push_pending(state: GameState, *, kind: PendingKind, player_id: str, payload: Dict[str, Any]) -> PendingAction:
    pa = PendingAction(kind=kind, playerId=player_id, payload=payload, id=_next_id(state))
    state.pending.append(pa)
    return pa


def _next_idx(state: GameState) -> int:
    n = len(state.players)
    return (state.current_idx + 1) % n


def _first_hidden(g: Grid) -> GridPos:
    for r in range(3):
        for c in range(3):
            if g.cells[r][c] is not None and not g.visible[r][c]:
                return (r, c)
    for r in range(3):
        for c in range(3):
            if g.cells[r][c] is not None:
                return (r, c)
    return (0, 0)


def _setup_find_next_reveal_pos(g: Grid) -> Optional[GridPos]:
    for r in range(3):
        for c in range(3):
            if g.cells[r][c] is not None and not g.visible[r][c]:
                return (r, c)
    return None


def step(state: GameState) -> None:
    """
    Progress game logic until either:
    - a new PendingAction is created, OR
    - the round ends (no pending).
    """
    if state.pending:
        return
    # Setup: two reveals per player
    if state._phase == "setup_reveals":
        for p in state.players:
            state._setup_reveals_done.setdefault(p.id, 0)
        for p in state.players:
            if state._setup_reveals_done.get(p.id, 0) < 2:
                pos = _setup_find_next_reveal_pos(p.grid)
                if pos is None:
                    state._setup_reveals_done[p.id] = 2
                    continue
                r, c = pos
                _push_pending(state, kind="reveal_card", player_id=p.id, payload={"context": "initial_reveal", "r": r, "c": c})
                return
        state._phase = "setup_discard"
    if state._phase == "setup_discard":
        _push_pending(state, kind="reveal_card", player_id=state.players[0].id, payload={"context": "initial_discard"})
        return
    # Playing
    if state._phase == "playing":
        if is_round_over(state):
            return
        p = _player(state)
        g = p.grid
        top = _top_discard(state)
        if p.kind == "H":
            hc = state._human_ctx
            if not hc:
                choices = ["draw"] if top is None else ["discard", "draw"]
                _push_pending(state, kind="choose_action", player_id=p.id, payload={"allowed": choices, "context": "turn_action"})
                return
            act = hc.get("action")
            if act == "discard":
                if "target" not in hc:
                    _push_pending(state, kind="choose_pos", player_id=p.id, payload={"context": "take_discard_target"})
                    return
                tr, tc = cast(GridPos, hc["target"])
                if not g.visible[tr][tc]:
                    _push_pending(state, kind="replace_hidden_card", player_id=p.id, payload={"r": tr, "c": tc})
                    return
                apply_human_action(state, action="take_discard", target=(tr, tc))
                maybe_trigger_end(state)
                state._human_ctx = {}
                if not state.pending:
                    state.current_idx = _next_idx(state)
                return
            if act == "draw":
                if "drawn" not in hc:
                    _push_pending(state, kind="reveal_card", player_id=p.id, payload={"context": "drawn"})
                    return
                if "after_draw_choice" not in hc:
                    _push_pending(state, kind="choose_action", player_id=p.id, payload={"allowed": ["place", "discard"], "context": "after_draw"})
                    return
                if hc["after_draw_choice"] == "place":
                    if "target" not in hc:
                        _push_pending(state, kind="choose_pos", player_id=p.id, payload={"context": "place_target"})
                        return
                    tr, tc = cast(GridPos, hc["target"])
                    if not g.visible[tr][tc]:
                        _push_pending(state, kind="replace_hidden_card", player_id=p.id, payload={"r": tr, "c": tc})
                        return
                    apply_human_action(state, action="draw_place", target=(tr, tc), drawn=cast(Card, hc["drawn"]))
                    maybe_trigger_end(state)
                    state._human_ctx = {}
                    if not state.pending:
                        state.current_idx = _next_idx(state)
                    return
                else:
                    if "flip_pos" not in hc:
                        _push_pending(state, kind="choose_pos", player_id=p.id, payload={"context": "flip_target_hidden"})
                        return
                    fr, fc = cast(GridPos, hc["flip_pos"])
                    _push_pending(state, kind="reveal_card", player_id=p.id, payload={"context": "flip_reveal", "r": fr, "c": fc})
                    return
        else:  # AI
            ac = state._ai_ctx
            if not ac:
                action, kwargs, info = ai_decide(state)
                state.explain = info
                ac.update({"decided": action, "kwargs": kwargs})
            action = cast(str, state._ai_ctx.get("decided"))
            if action == "take_discard":
                target = cast(GridPos, state._ai_ctx["kwargs"].get("target"))
                r, c = target
                if not g.visible[r][c]:
                    _push_pending(state, kind="ai_reveal_needed", player_id=p.id, payload={"what": "replaced_hidden", "r": r, "c": c})
                    return
                ai_apply(state, "take_discard", target=target)
                state._ai_ctx = {}
                maybe_trigger_end(state)
                if not state.pending:
                    state.current_idx = _next_idx(state)
                return
            else:  # draw path
                if "drawn" not in ac:
                    _push_pending(state, kind="ai_reveal_needed", player_id=p.id, payload={"what": "drawn"})
                    return
                drawn = cast(Card, ac["drawn"])  # type: ignore[assignment]
                (score, pos), best_cand, _cands = evaluate_best_place(
                    g,
                    drawn,
                    state.deck,
                    state.cfg.ai_depth,
                    state.cfg.ai_mc_samples,
                    state.cfg.ai_mc_seed,
                    top_discard=top,
                )
                r, c = pos
                can_flip = g.count_hidden() > 0
                exp_draw = state.deck.expected_value_unseen()
                ev_after_draw_place = best_cand.baseline + best_cand.ev_delta
                ev_after_draw_discard = float(g.known_sum()) + (exp_draw if can_flip else 0.0)
                place = (not can_flip) or (ev_after_draw_place <= ev_after_draw_discard)
                if place:
                    if not g.visible[r][c]:
                        _push_pending(state, kind="ai_reveal_needed", player_id=p.id, payload={"what": "replaced_hidden", "r": r, "c": c})
                        state._ai_ctx["target"] = (r, c)
                        return
                    ai_apply(state, "draw_place", drawn=drawn, target=(r, c))
                    state._ai_ctx = {}
                    maybe_trigger_end(state)
                    if not state.pending:
                        state.current_idx = _next_idx(state)
                    return
                else:
                    fr, fc = _first_hidden(g)
                    _push_pending(state, kind="ai_reveal_needed", player_id=p.id, payload={"what": "flipped_card", "r": fr, "c": fc, "drawn": {"c": drawn.c, "v": drawn.v}})
                    state._ai_ctx["flip_pos"] = (fr, fc)
                    return


def resolve(state: GameState, actionId: str, response: Dict[str, Any]) -> None:
    """
    Consume a PendingAction by ID, apply the given response to game state,
    clear it from pending, and continue progression (via step).
    """
    idx = next((i for i, a in enumerate(state.pending) if a.id == actionId), -1)
    assert idx >= 0, "Pending action not found"
    pa = state.pending.pop(idx)
    kind = pa.kind
    pid = pa.playerId
    p = next(p for p in state.players if p.id == pid)
    g = p.grid

    if kind == "reveal_card":
        col = response.get("c")
        val = response.get("v")
        assert isinstance(col, str) and isinstance(val, int)
        card = Card(col, int(val))
        ctx = pa.payload.get("context")
        if ctx == "initial_reveal":
            rr = int(pa.payload["r"])  # provided by step payload
            cc = int(pa.payload["c"])  # provided by step payload
            state.deck.see_card(card)
            g.reveal_known(rr, cc, card)
            state._setup_reveals_done[pid] = state._setup_reveals_done.get(pid, 0) + 1
        elif ctx == "initial_discard":
            state.discard = []
            _put_on_discard(state, [card])
            _append_log(state, f"INIT_DISCARD: {card}")
            # Determine starting player by lowest two sum
            # Will be computed after returning to step()
        elif ctx == "drawn":
            state.deck.see_card(card)
            state._human_ctx["drawn"] = card
        elif ctx == "flip_reveal":
            rr = int(pa.payload["r"])  # provided by step payload
            cc = int(pa.payload["c"])  # provided by step payload
            state.deck.see_card(card)
            g.reveal_known(rr, cc, card)
            draw_card = cast(Optional[Card], state._human_ctx.get("drawn"))
            assert draw_card is not None
            apply_human_action(state, action="draw_discard_flip", drawn=draw_card, flipped=(rr, cc), flipped_card=card)
            state._human_ctx = {}
    elif kind == "replace_hidden_card":
        r = int(pa.payload["r"])  # provided by step payload
        c = int(pa.payload["c"])  # provided by step payload
        col = response.get("c")
        val = response.get("v")
        assert isinstance(col, str) and isinstance(val, int)
        card = Card(col, int(val))
        state.deck.see_card(card)
        hc = state._human_ctx
        if hc.get("action") == "discard":
            apply_human_action(state, action="take_discard", target=(r, c), replaced_hidden=card)
            state._human_ctx = {}
        else:
            drawn = cast(Optional[Card], hc.get("drawn"))
            assert drawn is not None
            apply_human_action(state, action="draw_place", target=(r, c), drawn=drawn, replaced_hidden=card)
            state._human_ctx = {}
    elif kind == "choose_action":
        choice = str(response["choice"])  # required
        ctx = pa.payload.get("context")
        if ctx == "turn_action":
            state._human_ctx = {"action": choice}
        elif ctx == "after_draw":
            state._human_ctx["after_draw_choice"] = choice
    elif kind == "choose_pos":
        r = int(response["row"])  # required
        c = int(response["col"])  # required
        ctx = pa.payload.get("context")
        if ctx in ("take_discard_target", "place_target"):
            state._human_ctx["target"] = (r, c)
        elif ctx == "flip_target_hidden":
            state._human_ctx["flip_pos"] = (r, c)
    elif kind == "ai_reveal_needed":
        what = str(pa.payload.get("what"))
        col = response.get("c")
        val = response.get("v")
        assert isinstance(col, str) and isinstance(val, int)
        card = Card(col, int(val))
        state.deck.see_card(card)
        ac = state._ai_ctx
        if what == "drawn":
            ac["drawn"] = card
        elif what == "replaced_hidden":
            ac["replaced_hidden"] = card
            decided = cast(str, ac.get("decided", ""))
            if decided == "take_discard":
                target = cast(GridPos, ac.get("kwargs", {}).get("target"))
                ai_apply(state, "take_discard", target=target, replaced_hidden=card)
                state._ai_ctx = {}
            elif decided == "draw":
                tgt = cast(Optional[GridPos], ac.get("target"))
                if tgt is not None:
                    ai_apply(state, "draw_place", drawn=cast(Card, ac.get("drawn")), target=tgt, replaced_hidden=card)
                    state._ai_ctx = {}
        elif what == "flipped_card":
            rr = int(pa.payload["r"])  # provided by step payload
            cc = int(pa.payload["c"])  # provided by step payload
            drawn = cast(Card, ac.get("drawn"))
            ai_apply(state, "draw_discard_flip", drawn=drawn, flipped=(rr, cc), flipped_card=card)
            state._ai_ctx = {}
    elif kind == "choose_diag_compact":
        choice = str(response["choice"])  # required
        assert choice in ("vertical", "horizontal")
        g.compact_after_diag(cast(Literal["vertical", "horizontal"], choice))
        _append_log(state, f"HILO_COMPACT: diag -> {choice}")

    # After resolving, potentially advance phase
    if state._phase == "setup_discard" and not state.pending:
        # Compute starting player by lowest two visible sum
        start_sums: List[Tuple[int, int]] = []
        for idx, pp in enumerate(state.players):
            cnt = 0
            ssum = 0
            for r in range(3):
                for c in range(3):
                    if pp.grid.cells[r][c] is not None and pp.grid.visible[r][c]:
                        ssum += cast(Card, pp.grid.cells[r][c]).v
                        cnt += 1
                        if cnt >= 2:
                            break
                if cnt >= 2:
                    break
            start_sums.append((ssum, idx))
        state.current_idx = min(start_sums, key=lambda t: t[0])[1] if start_sums else 0
        state._phase = "playing"

    # Continue progression
    step(state)
