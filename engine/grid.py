from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Literal

from .types import Card, GridPos


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


# --- Pure helpers for chain resolution (no game state, no I/O) ---

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

