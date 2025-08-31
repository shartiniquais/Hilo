from __future__ import annotations

from typing import Dict, Optional, List
import random

from .types import Card, COLOR_MAP


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
            for v, k in self.counts[c].items():
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

