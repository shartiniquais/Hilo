from __future__ import annotations

from typing import Any, Dict, List, Tuple, Literal
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    NewRoundReq,
    StepReq,
    ResolveReq,
    GetStateResp,
    StateEnvelope,
)

# Engine import (kept via engine.core as requested)
from engine.core import (
    GameConfig,
    GameState,
    new_game,
    step as engine_step,
    resolve as engine_resolve,
    to_json,
    COLOR_MAP,
)


# In-memory session store
SESSIONS: Dict[str, GameState] = {}


def _new_session_id() -> str:
    return uuid.uuid4().hex


def get_state(session_id: str) -> GameState:
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def save_state(session_id: str, state: GameState) -> None:
    SESSIONS[session_id] = state


def _validate_response_payload(payload: Dict[str, Any]) -> None:
    # Best-effort lightweight guards (engine also asserts)
    if "row" in payload:
        r = payload.get("row")
        if not isinstance(r, int) or r < 0 or r > 2:
            raise HTTPException(status_code=422, detail="row must be 0..2")
    if "col" in payload:
        c = payload.get("col")
        if not isinstance(c, int) or c < 0 or c > 2:
            raise HTTPException(status_code=422, detail="col must be 0..2")
    if "v" in payload:
        v = payload.get("v")
        if not isinstance(v, int) or v < -1 or v > 11:
            raise HTTPException(status_code=422, detail="v must be -1..11")
    if "c" in payload:
        cc = payload.get("c")
        if not isinstance(cc, str) or cc not in COLOR_MAP:
            raise HTTPException(status_code=422, detail="c must be one of R,B,G,Y,P,O,L,M")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/new-round", response_model=StateEnvelope)
def new_round(req: NewRoundReq) -> StateEnvelope:
    try:
        # Map players to engine config format
        players_cfg: List[Tuple[Literal["H", "AI"], str]] = []
        for p in req.players:
            kind = p.get("kind")
            name = p.get("name")
            if kind not in ("H", "AI") or not isinstance(name, str) or not name:
                raise HTTPException(status_code=422, detail="Invalid players specification")
            players_cfg.append((kind, name))  # type: ignore[arg-type]

        cfg = GameConfig(
            players=players_cfg,  # type: ignore[arg-type]
            ai_depth=req.aiDepth,
            ai_mc_samples=int(req.aiMcSamples),
            ai_mc_seed=int(req.aiMcSeed),
            color_map=dict(COLOR_MAP),
            score_limit=int(req.scoreLimit),
        )
        state = new_game(cfg)
        # Kick off initial pending
        engine_step(state)
        sid = _new_session_id()
        save_state(sid, state)
        return StateEnvelope(sessionId=sid, state=to_json(state))
    except HTTPException:
        raise
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"new-round failed: {e}")


@app.get("/state/{sessionId}", response_model=GetStateResp)
def get_state_endpoint(sessionId: str) -> GetStateResp:
    state = get_state(sessionId)
    return GetStateResp(state=to_json(state))


@app.post("/step", response_model=GetStateResp)
def step_endpoint(req: StepReq) -> GetStateResp:
    try:
        state = get_state(req.sessionId)
        engine_step(state)
        save_state(req.sessionId, state)
        return GetStateResp(state=to_json(state))
    except HTTPException:
        raise
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"step failed: {e}")


@app.post("/resolve", response_model=GetStateResp)
def resolve_endpoint(req: ResolveReq) -> GetStateResp:
    try:
        state = get_state(req.sessionId)
        # lightweight validation on payload shape
        _validate_response_payload(req.response)
        engine_resolve(state, req.actionId, req.response)
        # resolve() internally calls step(), but not guaranteed everywhere; one more step is harmless if no pending
        save_state(req.sessionId, state)
        return GetStateResp(state=to_json(state))
    except HTTPException:
        raise
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"resolve failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

