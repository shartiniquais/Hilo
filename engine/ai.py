from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
import random

from .types import Card, GridPos
from .grid import Grid, apply_hilo_chain_pure
from .deck import DeckTracker


@dataclass
class CandidateEval:
    r: int
    c: int
    kind: Literal["visible", "hidden"]
    result_score: float
    delta_score: float
    p_improve: float
    ev_delta_hidden: Optional[float]
    immediate_hilo: bool
    removed_sum: int
    v_known: Optional[int]
    baseline: float
    ev_delta: float


def _lines_through(r: int, c: int) -> List[List[GridPos]]:
    pos = (r, c)
    return [line for line in Grid.lines() if pos in line]


def _state_signature(g: Grid, card: Card, pos: GridPos, top_discard: Optional[Card], deck: DeckTracker) -> int:
    parts: List[str] = []
    for r in range(3):
        for c in range(3):
            cell = g.cells[r][c]
            vis = g.visible[r][c]
            if cell is None:
                parts.append("..")
            elif vis:
                parts.append(f"{cell.c}{cell.v}")
            else:
                parts.append("??")
    td = top_discard
    parts.append(f"TD:{td.c}{td.v}" if td is not None else "TD:None")
    parts.append(f"CAND:{card.c}{card.v}@{pos[0]}{pos[1]}")
    colors_dist = deck.remaining_colors_distribution()
    parts.extend([f"{k}:{v}" for k, v in sorted(colors_dist.items())])
    return hash("|".join(parts))


def _expected_hilo_bonus(deck: DeckTracker, g: Grid, r: int, c: int, card: Card) -> float:
    # Expected value of a potential future HILO (color match) involving (r,c)
    total = deck.remaining_total()
    if total <= 0:
        return 0.0
    colors_dist = deck.remaining_colors_distribution()
    count_c = colors_dist.get(card.c, 0)
    if count_c <= 0:
        return 0.0
    e_val_color = deck.expected_value_for_color(card.c)
    bonus = 0.0
    for line in _lines_through(r, c):
        other = [(rr, cc) for (rr, cc) in line if not (rr == r and cc == c)]
        blocked = False
        visible_same = 0
        visible_sum = 0
        hidden_cnt = 0
        for (rr, cc) in other:
            cell = g.cells[rr][cc]
            if cell is None:
                blocked = True
                break
            if g.visible[rr][cc]:
                if cell.c == card.c:
                    visible_same += 1
                    visible_sum += cell.v
                else:
                    blocked = True
                    break
            else:
                hidden_cnt += 1
        if blocked:
            continue
        # Deterministic case handled by simulation; only include uncertain cases
        if hidden_cnt == 1 and visible_same == 1:
            p = count_c / float(total)
            expected_removed = float(card.v + visible_sum) + e_val_color
            bonus += p * expected_removed
        elif hidden_cnt == 2 and visible_same == 0:
            if total >= 2 and count_c >= 2:
                p = (count_c / float(total)) * ((count_c - 1) / float(total - 1))
            else:
                p = 0.0
            expected_removed = float(card.v) + 2.0 * e_val_color
            bonus += p * expected_removed
    return bonus


def _mc_candidate_delta(
    ai_depth: int,
    ai_mc_samples: int,
    ai_mc_seed: int,
    deck: DeckTracker,
    base: Grid,
    card: Card,
    pos: GridPos,
    top_discard: Optional[Card],
) -> float:
    if ai_depth < 2:
        return 0.0
    # If nothing hidden, no need to sample
    hidden_before = 0
    for r in range(3):
        for c in range(3):
            if base.cells[r][c] is not None and not base.visible[r][c]:
                hidden_before += 1
    if hidden_before == 0:
        return 0.0
    sig = _state_signature(base, card, pos, top_discard, deck)
    rng = random.Random(ai_mc_seed ^ (sig & 0xFFFFFFFF))
    expected_hidden_sum_before = float(hidden_before) * deck.expected_value_unseen()
    known_before = base.known_sum()
    acc = 0.0
    samples = max(1, ai_mc_samples)
    for _ in range(samples):
        dt = deck.copy()
        sim = base.clone()
        # Apply the candidate placement and resolve deterministic chains
        rr, cc = pos
        sim.cells[rr][cc] = card
        sim.visible[rr][cc] = True
        apply_hilo_chain_pure(sim, discard=[], chooser=None)
        # Fill hidden with samples
        for r in range(3):
            for c in range(3):
                if sim.cells[r][c] is not None and not sim.visible[r][c]:
                    sampled = dt.sample_card_without_replacement(rng)
                    if sampled is None:
                        continue
                    sim.cells[r][c] = sampled
                    sim.visible[r][c] = True
        # Resolve potential chains from sampled info
        apply_hilo_chain_pure(sim, discard=[], chooser=None)
        sim_full = sim.full_sum()
        delta = float(sim_full) - float(known_before) - expected_hidden_sum_before
        acc += delta
    return acc / float(samples)


def evaluate_best_place(
    g: Grid,
    card: Card,
    deck: DeckTracker,
    ai_depth: int,
    ai_mc_samples: int,
    ai_mc_seed: int,
    top_discard: Optional[Card] = None,
) -> Tuple[Tuple[float, GridPos], CandidateEval, List[CandidateEval]]:
    current_known = g.known_sum()
    e_hidden = deck.expected_value_unseen()
    candidates: List[CandidateEval] = []
    best_score = float(10**9)
    best_pos: GridPos = (0, 0)
    best_cand: Optional[CandidateEval] = None
    for r in range(3):
        for c in range(3):
            if g.cells[r][c] is None:
                continue
            sim = g.clone()
            sim.cells[r][c] = card
            sim.visible[r][c] = True
            # Resolve deterministic chains with pure helper
            apply_hilo_chain_pure(sim, discard=[], chooser=None)
            after_known = sim.known_sum()
            visible_before = g.visible[r][c]
            baseline = float(current_known) if visible_before else float(current_known) + float(e_hidden)
            base_delta = float(after_known) - baseline
            # Color-aware expected HILO encouragement
            hilo_bonus = _expected_hilo_bonus(deck, g, r, c, card)
            base_delta -= float(hilo_bonus)
            # Deterministic Monte Carlo to refine estimate
            mc_delta = _mc_candidate_delta(ai_depth, ai_mc_samples, ai_mc_seed, deck, g, card, (r, c), top_discard)
            score = base_delta + mc_delta

            # Candidate features for explain mode
            kind: Literal["visible", "hidden"] = "visible" if visible_before else "hidden"
            v_known: Optional[int] = None
            if visible_before:
                cell = g.cells[r][c]
                assert cell is not None
                v_known = cell.v
                p_improve = 1.0 if card.v < cell.v else 0.0
                ev_delta_hidden = None
            else:
                p_improve = deck.p_value_gt(float(card.v))
                ev_delta_hidden = float(e_hidden) - float(card.v)

            # Immediate HILO detection and removed sum
            immediate_hilo = False
            removed_sum = 0
            for line in _lines_through(r, c):
                ok = True
                cards_line: List[Card] = []
                for (rr, cc) in line:
                    if rr == r and cc == c:
                        cards_line.append(card)
                        continue
                    cell0 = g.cells[rr][cc]
                    if cell0 is None or not g.visible[rr][cc]:
                        ok = False
                        break
                    cards_line.append(cell0)
                if not ok:
                    continue
                colors = {cd.c for cd in cards_line}
                if len(colors) == 1:
                    immediate_hilo = True
                    removed_sum = sum(cd.v for cd in cards_line)
                    break

            result_score = float(after_known)
            delta_score = float(current_known) - result_score
            cand = CandidateEval(
                r=r,
                c=c,
                kind=kind,
                result_score=result_score,
                delta_score=delta_score,
                p_improve=float(p_improve),
                ev_delta_hidden=ev_delta_hidden,
                immediate_hilo=immediate_hilo,
                removed_sum=int(removed_sum),
                v_known=v_known,
                baseline=float(baseline),
                ev_delta=float(score),
            )
            candidates.append(cand)

            if score < best_score or (abs(score - best_score) < 1e-9 and (r, c) < best_pos):
                best_score = score
                best_pos = (r, c)
                best_cand = cand

    assert best_cand is not None
    return (best_score, best_pos), best_cand, candidates

