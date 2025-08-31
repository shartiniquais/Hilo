from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple, TypeAlias

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

# Alias for all allowed pending kinds for better type reuse
PendingKind: TypeAlias = Literal[
    "choose_action",
    "choose_pos",
    "reveal_card",
    "replace_hidden_card",
    "ai_reveal_needed",
    "choose_hilo_index",
    "choose_diag_compact",
]


@dataclass(frozen=True)
class Card:
    c: str  # color letter
    v: int  # value (-1..11)

    def __str__(self) -> str:
        return f"{self.c}{self.v}"


# Engine-driven pending action descriptor for external resolution
@dataclass
class PendingAction:
    kind: PendingKind
    playerId: str
    payload: Dict[str, Any]
    id: str  # unique id for correlation
