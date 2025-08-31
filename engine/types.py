from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

# Color legend (keep Lime, not Cyan)
COLOR_MAP: Dict[str, str] = {
    "R": "Red",
    "B": "Blue",
    "G": "Green",
    "Y": "Yellow",
    "P": "Purple",
    "O": "Orange",
    "L": "Lime",
    "M": "Magenta",
}

GridPos = Tuple[int, int]
PlayerKind = Literal["H", "AI"]


@dataclass(frozen=True)
class Card:
    c: str  # color letter
    v: int  # value (-1..11)

    def __str__(self) -> str:
        return f"{self.c}{self.v}"

