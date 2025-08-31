from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# Domain primitives for JSON bridge
Color = Literal["R", "B", "G", "Y", "P", "O", "L", "M"]


class Card(BaseModel):
    c: Color
    v: int = Field(..., ge=-1, le=11)


class Pos(BaseModel):
    r: int = Field(..., ge=0, le=2)
    c: int = Field(..., ge=0, le=2)


class NewRoundReq(BaseModel):
    players: List[Dict[str, str]]
    aiDepth: Literal[1, 2] = 2
    aiMcSamples: int = 512
    aiMcSeed: int = 1337
    scoreLimit: int = 100


class StepReq(BaseModel):
    sessionId: str


class ResolveReq(BaseModel):
    sessionId: str
    actionId: str
    response: Dict[str, Any]


class GetStateResp(BaseModel):
    state: Dict[str, Any]


class StateEnvelope(BaseModel):
    sessionId: str
    state: Dict[str, Any]

