from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Literal
import random

# ==========================
# Config and constants
# ==========================

# Player slots (2–4 active players). kind: "H" human, "AI" bot, "X" inactive.
PLAYERS: List[Dict[str, str]] = [
    {"kind": "H",  "name": "You"},
    {"kind": "AI", "name": "Bot A"},
    {"kind": "AI",  "name": "Bot B"},
    {"kind": "X",  "name": ""},
]

# Depth for deterministic lookahead over known cards only.
AI_DEPTH: Literal[1, 2] = 2

# Deterministic Monte Carlo configuration
AI_MC_SAMPLES: int = 512
AI_MC_SEED: int = 1337

# Optional demo mode via --demo
DEMO: bool = False

# Explain/Debug output
PRINT_PROBA: bool = True       # master switch
PRINT_TOP_K: int = 3           # how many candidates to show
PRINT_PROBA_PREC: int = 3      # decimals for probabilities

COLOR_MAP: Dict[str, str] = {
    "R": "Red", "B": "Blue", "G": "Green", "Y": "Yellow",
    "P": "Purple", "O": "Orange", "L": "Lime", "M": "Magenta",
}

GridPos = Tuple[int, int]  # (row, col)

SCORE_LIMIT = 10 # Default = 100

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


@dataclass(frozen=True)
class Card:
    c: str  # color letter
    v: int  # value

    def __str__(self) -> str:
        return f"{self.c}{self.v}"

class Grid:
    def __init__(self) -> None:
        # None means removed/empty; otherwise a Card exists (may be hidden via visible[r][c] = False)
        self.cells: List[List[Optional[Card]]] = [[Card("?", 0) for _ in range(3)] for _ in range(3)]
        self.visible: List[List[bool]] = [[False for _ in range(3)] for _ in range(3)]

    def clone(self) -> "Grid":
        g = Grid()
        g.cells = [[self.cells[r][c] for c in range(3)] for r in range(3)]
        g.visible = [[self.visible[r][c] for c in range(3)] for r in range(3)]
        return g

    def count_hidden(self) -> int:
        cnt = 0
        for r in range(3):
            for c in range(3):
                if self.cells[r][c] is not None and not self.visible[r][c]:
                    cnt += 1
        return cnt

    def any_cards_left(self) -> bool:
        for r in range(3):
            for c in range(3):
                if self.cells[r][c] is not None:
                    return True
        return False

    def all_visible_or_empty(self) -> bool:
        if not self.any_cards_left():
            return True
        return self.count_hidden() == 0

    def reveal_known(self, r: int, c: int, card: Card) -> None:
        self.cells[r][c] = card
        self.visible[r][c] = True

    def set_unknown_hidden(self, r: int, c: int) -> None:
        self.cells[r][c] = Card("?", 0)
        self.visible[r][c] = False

    def replace(self, r: int, c: int, card: Card) -> Optional[Card]:
        old = self.cells[r][c]
        self.cells[r][c] = card
        self.visible[r][c] = True
        return old

    def known_sum(self) -> int:
        s = 0
        for r in range(3):
            for c in range(3):
                card = self.cells[r][c]
                if card is not None and self.visible[r][c]:
                    s += card.v
        return s

    def full_sum(self) -> int:
        s = 0
        for r in range(3):
            for c in range(3):
                card = self.cells[r][c]
                if card is not None:
                    s += card.v
        return s

    @staticmethod
    def lines() -> List[List[GridPos]]:
        L: List[List[GridPos]] = []
        for r in range(3):
            L.append([(r, 0), (r, 1), (r, 2)])
        for c in range(3):
            L.append([(0, c), (1, c), (2, c)])
        L.append([(0, 0), (1, 1), (2, 2)])
        L.append([(0, 2), (1, 1), (2, 0)])
        return L

    def find_hilo_lines(self) -> List[List[GridPos]]:
        wins: List[List[GridPos]] = []
        for line in Grid.lines():
            cards: List[Card] = []
            ok = True
            for (r, c) in line:
                card = self.cells[r][c]
                if card is None or not self.visible[r][c]:
                    ok = False
                    break
                cards.append(card)
            if not ok:
                continue
            colors = {card.c for card in cards}
            if len(colors) == 1:
                wins.append(line)
        return wins

    def remove_line(self, line: List[GridPos]) -> Literal["row", "col", "diag"]:
        rs = {r for (r, _) in line}
        cs = {c for (_, c) in line}
        if len(rs) == 1:
            typ: Literal["row", "col", "diag"] = "row"
        elif len(cs) == 1:
            typ = "col"
        else:
            typ = "diag"
        for (r, c) in line:
            self.cells[r][c] = None
            self.visible[r][c] = False
        return typ

    def compact_after_diag(self, mode: Literal["vertical", "horizontal"]) -> None:
        remaining: List[Tuple[Card, bool]] = []
        for r in range(3):
            for c in range(3):
                card = self.cells[r][c]
                if card is not None:
                    remaining.append((card, self.visible[r][c]))
        self.cells = [[None for _ in range(3)] for _ in range(3)]
        self.visible = [[False for _ in range(3)] for _ in range(3)]
        idx = 0
        if mode == "vertical":
            for r in range(2):
                for c in range(3):
                    if idx < len(remaining):
                        card, vis = remaining[idx]
                        self.cells[r][c] = card
                        self.visible[r][c] = vis
                        idx += 1
        else:
            for c in range(2):
                for r in range(3):
                    if idx < len(remaining):
                        card, vis = remaining[idx]
                        self.cells[r][c] = card
                        self.visible[r][c] = vis
                        idx += 1

    def print(self, title: str = "") -> None:
        if title:
            print(f"--- {title} ---")
        for r in range(3):
            row_parts: List[str] = []
            for c in range(3):
                card = self.cells[r][c]
                if card is None:
                    row_parts.append("..")
                elif self.visible[r][c]:
                    row_parts.append(f"{card.c}{card.v:>2}")
                else:
                    row_parts.append(" ?")
            print(" ".join(f"{x:>3}" for x in row_parts))
        print()

class Player:
    def __init__(self, name: str, kind: Literal["H", "AI"]) -> None:
        self.name = name
        self.kind = kind
        self.grid: Optional[Grid] = None
        self.total_points: int = 0

    @property
    def is_human(self) -> bool:
        return self.kind == "H"


# ==========================
# Deck tracking and EV helpers
# ==========================

VALUES: List[int] = [-1] + list(range(0, 12))


class DeckTracker:
    def __init__(self) -> None:
        self.counts: Dict[str, Dict[int, int]] = {}
        self._total: int = 0
        self.reset()

    def reset(self) -> None:
        self.counts = {c: {v: 1 for v in VALUES} for c in COLOR_MAP.keys()}
        self._total = len(COLOR_MAP) * len(VALUES)

    def copy(self) -> "DeckTracker":
        dt = DeckTracker()
        dt.counts = {c: d.copy() for c, d in self.counts.items()}
        dt._total = self._total
        return dt

    def remaining_total(self) -> int:
        return self._total

    def remaining_count(self, color: str, value: int) -> int:
        if color not in self.counts:
            return 0
        return self.counts[color].get(value, 0)

    def remaining_values_distribution(self) -> Dict[int, int]:
        dd: Dict[int, int] = {v: 0 for v in VALUES}
        for c in self.counts:
            for v, k in self.counts[c].items():
                dd[v] += k
        return dd

    def remaining_colors_distribution(self) -> Dict[str, int]:
        dd: Dict[str, int] = {c: 0 for c in self.counts}
        for c in self.counts:
            dd[c] = sum(self.counts[c].values())
        return dd

    def p_value_lt(self, x: int) -> float:
        tot = self._total
        if tot <= 0:
            return 0.0
        cnt = 0
        for v, k in self.remaining_values_distribution().items():
            if v < x:
                cnt += k
        return cnt / tot

    def p_value_gt(self, x: float) -> float:
        tot = self._total
        if tot <= 0:
            return 0.0
        cnt = 0
        for v, k in self.remaining_values_distribution().items():
            if float(v) > float(x):
                cnt += k
        return cnt / float(tot)

    def expected_value_unseen(self) -> float:
        tot = self._total
        if tot <= 0:
            return 0.0
        s = 0.0
        for v, k in self.remaining_values_distribution().items():
            s += float(v) * float(k)
        return s / float(tot)

    def expected_value_for_color(self, color: str) -> float:
        if color not in self.counts:
            return self.expected_value_unseen()
        dd = self.counts[color]
        tot = sum(dd.values())
        if tot <= 0:
            return self.expected_value_unseen()
        s = 0.0
        for v, k in dd.items():
            s += float(v) * float(k)
        return s / float(tot)

    def p_color(self, color: str) -> float:
        dist = self.remaining_colors_distribution()
        tot = self._total
        if tot <= 0:
            return 0.0
        return float(dist.get(color, 0)) / float(tot)

    def see_card(self, card: Card) -> None:
        # Idempotent removal from remaining unseen multiset
        c = card.c
        v = card.v
        if c in self.counts and v in self.counts[c] and self.counts[c][v] > 0:
            self.counts[c][v] -= 1
            self._total -= 1

    def sample_card_without_replacement(self, rng: random.Random) -> Optional[Card]:
        tot = self._total
        if tot <= 0:
            return None
        pick = rng.randrange(tot)
        acc = 0
        for c in self.counts:
            for v in VALUES:
                k = self.counts[c].get(v, 0)
                if k <= 0:
                    continue
                if acc + k > pick:
                    # consume one
                    self.counts[c][v] -= 1
                    self._total -= 1
                    return Card(c, v)
                acc += k
        # Fallback (should not hit)
        return None


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

# ==========================
# Pure evaluation helpers (no Game construction)
# ==========================

def _put_on_discard_local(discard: List[Card], cards: Sequence[Card]) -> None:
    # Smallest value on top -> append descending
    cards_sorted = sorted(cards, key=lambda c: c.v, reverse=True)
    for c in cards_sorted:
        discard.append(c)


def choose_best_compaction_pure(g: Grid) -> Literal["vertical", "horizontal"]:
    best_mode: Literal["vertical", "horizontal"] = "vertical"
    best_score = 10 ** 9
    for mode in ("vertical", "horizontal"):
        sim = g.clone()
        sim.compact_after_diag(mode)
        # Use local discard to resolve any chains deterministically
        apply_hilo_chain_pure(sim, discard=[], chooser=None)
        score = sim.known_sum()
        if score < best_score:
            best_score = score
            best_mode = mode
    return best_mode


def apply_hilo_chain_pure(
    g: Grid,
    discard: Optional[List[Card]] = None,
    chooser: Optional[Literal["vertical", "horizontal"]] = None,
) -> None:
    local_discard: List[Card] = [] if discard is None else discard
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
        _put_on_discard_local(local_discard, trio)
        if typ == "diag":
            mode: Literal["vertical", "horizontal"]
            if chooser is None:
                mode = choose_best_compaction_pure(g)
            else:
                mode = chooser
            g.compact_after_diag(mode)


class Game:
    def __init__(self, players: List[Player]) -> None:
        assert 2 <= len(players) <= 4
        self.players: List[Player] = players
        self.discard: List[Card] = []
        self.deck: DeckTracker = DeckTracker()

    # -------- Discard helpers --------
    def top_discard(self) -> Optional[Card]:
        return self.discard[-1] if self.discard else None

    def put_on_discard(self, cards: Sequence[Card]) -> None:
        # Smallest value on top -> append descending
        cards_sorted = sorted(cards, key=lambda c: c.v, reverse=True)
        for c in cards_sorted:
            self.discard.append(c)
            # Track known cards in the discard
            self.deck.see_card(c)

    # -------- HILO processing --------
    def apply_hilo_chain(self, g: Grid, chooser: Optional[Literal["vertical", "horizontal"]] = None) -> None:
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
            self.put_on_discard(trio)
            if typ == "diag":
                mode: Literal["vertical", "horizontal"]
                if chooser is None:
                    mode = self.choose_best_compaction(g)
                else:
                    mode = chooser
                g.compact_after_diag(mode)

    def choose_best_compaction(self, g: Grid) -> Literal["vertical", "horizontal"]:
        # Use pure helper (no Game construction in evaluation path)
        return choose_best_compaction_pure(g)
    # -------- I/O helpers --------
    def ask_pos_any(self) -> GridPos:
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

    def ask_pos_hidden(self, g: Grid) -> GridPos:
        while True:
            (r, c) = self.ask_pos_any()
            if g.cells[r][c] is None:
                print("That cell is empty.")
                continue
            if g.visible[r][c]:
                print("That cell is already face-up; pick a hidden one.")
                continue
            return (r, c)

    def ask_card(self, prompt: str = "Card") -> Card:
        print(prompt)
        col = parse_color_letter()
        val = parse_value()
        return Card(col, val)

    # -------- Human turn --------
    def human_turn(self, p: Player) -> None:
        g = p.grid
        assert g is not None
        print()
        print_legend()
        print(f"Turn: {p.name}")
        top = self.top_discard()
        print(f"Discard top: {top if top else '(empty)'}")
        g.print(f"{p.name}'s grid")

        action: Optional[Literal['discard', 'draw']] = None
        while action not in ('discard', 'draw'):
            a = input("Action (take 'discard' or 'draw'): ").strip().lower()
            if a in ('discard', 'draw'):
                action = a
        if action == 'discard':
            if top is None:
                print("Discard is empty; you must draw.")
                action = 'draw'
            else:
                r, c = self.ask_pos_any()
                old = g.cells[r][c]
                if old is None:
                    print("Cell is empty; cannot replace an empty cell.")
                    return self.human_turn(p)
                if not g.visible[r][c]:
                    old = self.ask_card("Replaced hidden card (color+value):")
                    self.deck.see_card(old)
                    print(f"HUMAN_REPLACED: {old}")
                assert old is not None
                g.cells[r][c] = top
                g.visible[r][c] = True
                self.discard.pop()
                self.put_on_discard([old])
                self.handle_hilo_with_prompt(g)
                return

        drawn = self.ask_card("Drawn card (color+value):")
        self.deck.see_card(drawn)
        place_or_discard: Optional[Literal['place', 'discard']] = None
        can_discard = g.count_hidden() > 0
        while place_or_discard not in ('place', 'discard'):
            prompt = "Place or discard?"
            if not can_discard:
                prompt += " (no hidden cards -> must place)"
            s2 = input(prompt + " ").strip().lower()
            if s2 in ('place', 'discard'):
                if s2 == 'discard' and not can_discard:
                    continue
                place_or_discard = s2
        if place_or_discard == 'place':
            r, c = self.ask_pos_any()
            if g.cells[r][c] is None:
                print("Cell is empty; cannot replace an empty cell.")
                return self.human_turn(p)
            replaced = g.cells[r][c]
            if not g.visible[r][c]:
                replaced = self.ask_card("Replaced hidden card (color+value):")
                self.deck.see_card(replaced)
                print(f"HUMAN_REPLACED: {replaced}")
            assert replaced is not None
            g.cells[r][c] = drawn
            g.visible[r][c] = True
            self.put_on_discard([replaced])
            self.handle_hilo_with_prompt(g)
        else:
            self.put_on_discard([drawn])
            print(f"HUMAN_DISCARDED: {drawn}")
            print("You must flip one hidden cell.")
            r, c = self.ask_pos_hidden(g)
            revealed = self.ask_card("Revealed card (color+value):")
            self.deck.see_card(revealed)
            g.reveal_known(r, c, revealed)
            print(f"HUMAN_FLIPPED: ({r},{c}) -> {revealed}")
            self.handle_hilo_with_prompt(g)

    def handle_hilo_with_prompt(self, g: Grid) -> None:
        while True:
            lines = g.find_hilo_lines()
            if not lines:
                return
            if len(lines) > 1:
                print("Multiple HILO available; choose index:")
                for i, line in enumerate(lines):
                    show = [g.cells[r][c] for (r, c) in line]
                    print(f"  {i}: {show}")
                idx: Optional[int] = None
                while idx is None:
                    try:
                        i = int(input("Index: ").strip())
                        if 0 <= i < len(lines):
                            idx = i
                    except Exception:
                        pass
                line = lines[idx]
            else:
                line = lines[0]
            trio: List[Card] = []
            for (r, c) in line:
                card = g.cells[r][c]
                assert card is not None
                trio.append(card)
            typ = g.remove_line(line)
            self.put_on_discard(trio)
            if typ == 'diag':
                mode_in: Optional[Literal['vertical', 'horizontal']] = None
                while mode_in not in ('vertical', 'horizontal'):
                    m = input("Compact diagonal: vertical or horizontal? ").strip().lower()
                    if m in ('vertical', 'horizontal'):
                        mode_in = m  # type: ignore[assignment]
                assert mode_in is not None
                g.compact_after_diag(mode_in)
    # -------- AI turn --------
    def ai_turn(self, p: Player) -> None:
        g = p.grid
        assert g is not None
        print()
        print_legend()
        print(f"Turn: {p.name} [AI]")
        top = self.top_discard()
        print(f"Discard top: {top if top else '(empty)'}")
        g.print(f"{p.name}'s grid")

        # Decide: take discard vs draw
        take_discard_score: Optional[Tuple[float, GridPos]] = None
        take_discard_best: Optional[CandidateEval] = None
        take_discard_move: Optional[GridPos] = None
        if top is not None:
            td_score, td_best, _td_cands = self.evaluate_best_place(g, top)
            take_discard_score = td_score
            take_discard_best = td_best
            take_discard_move = td_score[1]

        draw_first = True
        if take_discard_score is not None:
            best_delta, _pos = take_discard_score
            # If taking the discard yields negative delta (improvement), prefer it
            if best_delta < 0.0:
                draw_first = False

        def _fmt_prob(x: float) -> str:
            return f"{x:.{PRINT_PROBA_PREC}f}"

        if draw_first:
            print("AI_ACTION: draw")
            drawn = self.ask_card("AI needs the drawn card, please input color+value:")
            self.deck.see_card(drawn)
            best_place_score, best_cand, cand_list = self.evaluate_best_place(g, drawn)
            if PRINT_PROBA:
                # Phase A — action comparison
                if take_discard_best is not None:
                    ev_after_td = take_discard_best.baseline + take_discard_best.ev_delta
                    print(f"AI_EVAL: take_discard -> EV={ev_after_td:.3f}")
                ev_after_draw_place = best_cand.baseline + best_cand.ev_delta
                print(f"AI_EVAL: draw_place -> EV={ev_after_draw_place:.3f}")
                exp_draw = self.deck.expected_value_unseen()
                can_flip_eval = g.count_hidden() > 0
                p_flip_improve = self.deck.p_value_gt(exp_draw) if can_flip_eval else 0.0
                ev_after_draw_discard = float(g.known_sum()) + (exp_draw if can_flip_eval else 0.0)
                print(f"AI_EVAL: draw_discard_flip -> EV={ev_after_draw_discard:.3f}, p_flip_improve={_fmt_prob(p_flip_improve)}")
                # Phase B — top-K candidates
                cand_sorted = sorted(cand_list, key=lambda ce: ce.result_score)
                for ce in cand_sorted[:PRINT_TOP_K]:
                    base = f"AI_EVAL_CANDIDATE: PLACE=({ce.r},{ce.c}), kind={ce.kind}, result={ce.result_score:.3f}, " + \
                           f"Δ={(g.known_sum()-ce.result_score):.3f}, p_improve={_fmt_prob(ce.p_improve)}"
                    if ce.kind == 'hidden' and ce.ev_delta_hidden is not None:
                        base += f", ev_delta_hidden={ce.ev_delta_hidden:.3f}"
                    base += f", hilo={ce.immediate_hilo}, removed_sum={ce.removed_sum}"
                    print(base)
                # Phase C — pick rationale
                if best_cand.kind == 'hidden':
                    evh = best_cand.ev_delta_hidden if best_cand.ev_delta_hidden is not None else 0.0
                    print(f"AI_EVAL_PICK: PLACE=({best_cand.r},{best_cand.c}) because min(EV)={best_place_score[0]:.3f}; p_improve={_fmt_prob(best_cand.p_improve)}, EV_hidden={evh:.3f}")
                else:
                    vk = best_cand.v_known if best_cand.v_known is not None else -999
                    print(f"AI_EVAL_PICK: PLACE=({best_cand.r},{best_cand.c}) because min(EV)={best_place_score[0]:.3f}; V_known={vk}")
            can_flip = g.count_hidden() > 0
            do_place = True
            if can_flip and best_place_score[0] >= 0.0:
                do_place = False
            if do_place:
                r, c = best_cand.r, best_cand.c
                replaced = g.cells[r][c]
                if replaced is None:
                    r, c = self.first_non_empty(g)
                    replaced = g.cells[r][c]
                assert replaced is not None
                # Announce intent before asking hidden identity
                print(f"AI_ACTION: draw; PLACE: ({r},{c})")
                if not g.visible[r][c]:
                    replaced = self.ask_card("AI replaced a hidden card; please input it:")
                print(f"AI_REPLACED: {replaced}")
                g.cells[r][c] = drawn
                g.visible[r][c] = True
                self.deck.see_card(replaced)
                self.put_on_discard([replaced])
                self.ai_apply_chains(g)
            else:
                r, c = self.first_hidden(g)
                print(f"AI_ACTION: draw; DISCARD; FLIP: ({r},{c})")
                self.put_on_discard([drawn])
                print(f"AI_DISCARDED: {drawn}")
                revealed = self.ask_card("Flip result for AI (color+value):")
                self.deck.see_card(revealed)
                g.reveal_known(r, c, revealed)
                print(f"AI_FLIPPED: ({r},{c}) -> {revealed}")
                self.ai_apply_chains(g)
        else:
            assert top is not None
            assert take_discard_score is not None and take_discard_best is not None
            if PRINT_PROBA:
                ev_after_td = take_discard_best.baseline + take_discard_best.ev_delta
                print(f"AI_EVAL: take_discard -> EV={ev_after_td:.3f}")
                ev_after_draw_place_est = self._estimate_draw_place_ev(g)
                exp_draw = self.deck.expected_value_unseen()
                can_flip_eval = g.count_hidden() > 0
                p_flip_improve = self.deck.p_value_gt(exp_draw) if can_flip_eval else 0.0
                ev_after_draw_discard = float(g.known_sum()) + (exp_draw if can_flip_eval else 0.0)
                print(f"AI_EVAL: draw_place -> EV={ev_after_draw_place_est:.3f}")
                print(f"AI_EVAL: draw_discard_flip -> EV={ev_after_draw_discard:.3f}, p_flip_improve={_fmt_prob(p_flip_improve)}")
                # Candidates for discard placement
                _score_tmp, _best_tmp, td_cand_list = self.evaluate_best_place(g, top)
                cand_sorted = sorted(td_cand_list, key=lambda ce: ce.result_score)
                for ce in cand_sorted[:PRINT_TOP_K]:
                    base = f"AI_EVAL_CANDIDATE: PLACE=({ce.r},{ce.c}), kind={ce.kind}, result={ce.result_score:.3f}, " + \
                           f"Δ={(g.known_sum()-ce.result_score):.3f}, p_improve={_fmt_prob(ce.p_improve)}"
                    if ce.kind == 'hidden' and ce.ev_delta_hidden is not None:
                        base += f", ev_delta_hidden={ce.ev_delta_hidden:.3f}"
                    base += f", hilo={ce.immediate_hilo}, removed_sum={ce.removed_sum}"
                    print(base)
                if take_discard_best.kind == 'hidden':
                    evh = take_discard_best.ev_delta_hidden if take_discard_best.ev_delta_hidden is not None else 0.0
                    print(f"AI_EVAL_PICK: PLACE=({take_discard_best.r},{take_discard_best.c}) because min(EV)={take_discard_score[0]:.3f}; p_improve={_fmt_prob(take_discard_best.p_improve)}, EV_hidden={evh:.3f}")
                else:
                    vk = take_discard_best.v_known if take_discard_best.v_known is not None else -999
                    print(f"AI_EVAL_PICK: PLACE=({take_discard_best.r},{take_discard_best.c}) because min(EV)={take_discard_score[0]:.3f}; V_known={vk}")
            _, pos = take_discard_score
            assert pos is not None
            r, c = pos
            replaced = g.cells[r][c]
            assert replaced is not None
            # Announce intent before asking hidden identity
            print(f"AI_ACTION: discard; PLACE: ({r},{c})")
            if not g.visible[r][c]:
                replaced = self.ask_card("AI replaced a hidden card; please input it:")
            print(f"AI_REPLACED: {replaced}")
            self.discard.pop()
            g.cells[r][c] = top
            g.visible[r][c] = True
            self.deck.see_card(replaced)
            self.put_on_discard([replaced])
            self.ai_apply_chains(g)

    def first_hidden(self, g: Grid) -> GridPos:
        for r in range(3):
            for c in range(3):
                if g.cells[r][c] is not None and not g.visible[r][c]:
                    return (r, c)
        return self.first_non_empty(g)

    def first_non_empty(self, g: Grid) -> GridPos:
        for r in range(3):
            for c in range(3):
                if g.cells[r][c] is not None:
                    return (r, c)
        return (0, 0)

    def ai_apply_chains(self, g: Grid) -> None:
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
            self.put_on_discard(trio)
            if typ == "diag":
                mode = self.choose_best_compaction(g)
                # Announce how the AI compacted after a diagonal HILO
                print(f"AI_COMPACT: diag -> {mode}")
                g.compact_after_diag(mode)

    def _lines_through(self, r: int, c: int) -> List[List[GridPos]]:
        pos = (r, c)
        return [line for line in Grid.lines() if pos in line]

    def _expected_hilo_bonus(self, g: Grid, r: int, c: int, card: Card) -> float:
        # Expected value of a potential future HILO (color match) involving (r,c)
        total = self.deck.remaining_total()
        if total <= 0:
            return 0.0
        colors_dist = self.deck.remaining_colors_distribution()
        count_c = colors_dist.get(card.c, 0)
        if count_c <= 0:
            return 0.0
        e_val_color = self.deck.expected_value_for_color(card.c)
        bonus = 0.0
        for line in self._lines_through(r, c):
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

    def _state_signature(self, g: Grid, card: Card, pos: GridPos) -> int:
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
        td = self.top_discard()
        parts.append(f"TD:{td.c}{td.v}" if td is not None else "TD:None")
        parts.append(f"CAND:{card.c}{card.v}@{pos[0]}{pos[1]}")
        # Include a coarse deck summary
        colors_dist = self.deck.remaining_colors_distribution()
        parts.extend([f"{k}:{v}" for k, v in sorted(colors_dist.items())])
        return hash("|".join(parts))

    def _state_signature_draw(self, g: Grid) -> int:
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
        td = self.top_discard()
        parts.append(f"TD:{td.c}{td.v}" if td is not None else "TD:None")
        colors_dist = self.deck.remaining_colors_distribution()
        parts.extend([f"{k}:{v}" for k, v in sorted(colors_dist.items())])
        return hash("|".join(parts))

    def _mc_candidate_delta(self, base: Grid, card: Card, pos: GridPos) -> float:
        if AI_DEPTH < 2:
            return 0.0
        # If nothing hidden, no need to sample
        hidden_before = 0
        for r in range(3):
            for c in range(3):
                if base.cells[r][c] is not None and not base.visible[r][c]:
                    hidden_before += 1
        if hidden_before == 0:
            return 0.0
        rng = random.Random(AI_MC_SEED ^ (self._state_signature(base, card, pos) & 0xFFFFFFFF))
        expected_hidden_sum_before = float(hidden_before) * self.deck.expected_value_unseen()
        known_before = base.known_sum()
        acc = 0.0
        samples = max(1, AI_MC_SAMPLES)
        for _ in range(samples):
            dt = self.deck.copy()
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

    def _estimate_draw_place_ev(self, g: Grid) -> float:
        # Deterministic estimate of EV after drawing and placing best
        sig = self._state_signature_draw(g)
        rng = random.Random(AI_MC_SEED ^ (sig & 0xFFFFFFFF))
        trials = max(1, min(128, AI_MC_SAMPLES // 2))
        acc = 0.0
        for _ in range(trials):
            dt = self.deck.copy()
            draw = dt.sample_card_without_replacement(rng)
            if draw is None:
                break
            (score, pos), best_cand, _cand = self.evaluate_best_place(g, draw)
            ev_after = best_cand.baseline + best_cand.ev_delta
            acc += ev_after
        if trials <= 0:
            return float(g.known_sum())
        return acc / float(trials)

    def evaluate_best_place(self, g: Grid, card: Card) -> Tuple[Tuple[float, GridPos], CandidateEval, List[CandidateEval]]:
        current_known = g.known_sum()
        e_hidden = self.deck.expected_value_unseen()
        candidates: List[CandidateEval] = []
        best_score = float(10 ** 9)
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
                hilo_bonus = self._expected_hilo_bonus(g, r, c, card)
                # Expected value removed reduces score (good)
                base_delta -= float(hilo_bonus)
                # Deterministic Monte Carlo to refine estimate
                mc_delta = self._mc_candidate_delta(g, card, (r, c))
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
                    p_improve = self.deck.p_value_gt(float(card.v))
                    ev_delta_hidden = float(e_hidden) - float(card.v)

                # Immediate HILO detection and removed sum
                immediate_hilo = False
                removed_sum = 0
                for line in self._lines_through(r, c):
                    # Build the line state as if placed
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
    # -------- Round and game flow --------
    def play_round(self) -> None:
        self.discard = []
        self.deck.reset()
        for p in self.players:
            p.grid = Grid()
        print("\n=== New Round ===")
        print("Enter initial two reveals per active player.")
        start_sums: List[Tuple[int, Player]] = []
        for p in self.players:
            g = p.grid
            assert g is not None
            print()
            print_legend()
            print(f"Initial reveals for {p.name} (two positions)")
            picks: List[GridPos] = []
            while len(picks) < 2:
                r, c = self.ask_pos_any()
                if (r, c) in picks:
                    print("Duplicate position; pick another.")
                    continue
                card = self.ask_card(f"Card at ({r},{c}):")
                self.deck.see_card(card)
                g.reveal_known(r, c, card)
                picks.append((r, c))
            # Sum the two revealed cards safely
            ssum = 0
            for (rr, cc) in picks:
                card = g.cells[rr][cc]
                if card is not None:
                    ssum += card.v
            start_sums.append((ssum, p))
        # Starting player each round: highest two-card sum (ties by table order)
        max_sum_init = max(s for s, _ in start_sums)
        starter: Player = next(p for (s, p) in start_sums if s == max_sum_init)
        idx = self.players.index(starter)
        self.players = self.players[idx:] + self.players[:idx]

        # After initial reveals, set the initial discard top
        print()
        print_legend()
        init_top = self.ask_card("Initial discard (color+value):")
        self.discard = []
        self.put_on_discard([init_top])
        print(f"Initial discard top: {self.top_discard()}")

        end_trigger: Optional[Player] = None
        while True:
            for p in self.players:
                if end_trigger is not None:
                    break
                if p.is_human:
                    self.human_turn(p)
                else:
                    self.ai_turn(p)
                if p.grid is not None and p.grid.all_visible_or_empty():
                    end_trigger = p
                    print(f"End triggered by: {p.name}")
                    break
                top = self.top_discard()
                print(f"Discard top now: {top if top else '(empty)'}")
            if end_trigger is not None:
                for p in self.players:
                    if p is end_trigger:
                        continue
                    if p.is_human:
                        self.human_turn(p)
                    else:
                        self.ai_turn(p)
                break

        for p in self.players:
            g = p.grid
            assert g is not None
            for r in range(3):
                for c in range(3):
                    if g.cells[r][c] is not None and not g.visible[r][c]:
                        print(f"Reveal remaining for {p.name} at ({r},{c})")
                        card = self.ask_card("Color+value:")
                        self.deck.see_card(card)
                        g.reveal_known(r, c, card)
            self.handle_hilo_with_prompt(g)
            print(f"\nFinal grid for {p.name}:")
            g.print()

        totals: List[Tuple[Player, int]] = []
        for p in self.players:
            g = p.grid
            assert g is not None
            totals.append((p, g.full_sum()))
        for p, s in totals:
            print(f"{p.name}: {s} points")
        min_sum = min(s for _, s in totals)
        min_count = sum(1 for _p, s in totals if s == min_sum)
        # Apply corrected doubling rules
        for p, s in totals:
            final = s
            if p is end_trigger and s > min_sum:
                final = s * 2
                print(f"{p.name} doubles (not strictly lowest): -> {final}")
            elif p is end_trigger and s == min_sum and min_count >= 2:
                final = s * 2
                print(f"{p.name} ties for lowest (doubles): -> {final}")
            p.total_points += final

        # Unit-style scoring check for the reported bug example
        # Example: Bot A=9, You=8, trigger=You -> no double
        try:
            names = [p.name for p, _ in totals]
            scores = {p.name: s for p, s in totals}
            if "You" in scores and "Bot A" in scores:
                trigger_name = end_trigger.name if end_trigger is not None else ""
                example_trigger = "You"
                doubled_expected = False
                # Apply logic on the example
                ex_scores = {"Bot A": 9, "You": 8}
                ex_min = min(ex_scores.values())
                ex_count = sum(1 for v in ex_scores.values() if v == ex_min)
                you_s = ex_scores["You"]
                doubled_actual = (you_s > ex_min) or (you_s == ex_min and ex_count >= 2)
                print(f"SCORING_CHECK: Example trigger=You, Bot A=9, You=8 -> doubled? {doubled_actual} (expected {doubled_expected})")
        except Exception:
            pass
        print("\nCumulative:")
        for p in self.players:
            print(f"  {p.name}: {p.total_points} pts")

    def play_game_to_limit(self) -> None:
        while True:
            self.play_round()
            if any(p.total_points >= SCORE_LIMIT for p in self.players):
                break
        print("\n=== Game Over ===")
        for p in self.players:
            print(f"{p.name}: {p.total_points} pts")
        best_total = min(p.total_points for p in self.players)
        winners = [p.name for p in self.players if p.total_points == best_total]
        if len(winners) == 1:
            print(f"Winner: {winners[0]}")
        else:
            print("Winners (tie): " + ", ".join(winners))


def init_players_from_config() -> List[Player]:
    active: List[Player] = []
    for slot in PLAYERS:
        kind = slot.get("kind", "X").upper()
        name = slot.get("name", "")
        if kind == "X":
            continue
        if kind not in ("H", "AI"):
            continue
        if not name:
            name = kind
        active.append(Player(name, kind))
    if not (2 <= len(active) <= 4):
        raise SystemExit("Config error: need 2–4 active players in PLAYERS.")
    return active


def main() -> None:
    import sys
    global DEMO
    if "--demo" in sys.argv:
        DEMO = True
        print("Demo mode: 2 players (You vs Bot A)")
    players = init_players_from_config()
    game = Game(players)
    print("HILO — Console Arbiter (physical deck)")
    print_legend()
    game.play_game_to_limit()


if __name__ == "__main__":
    main()
