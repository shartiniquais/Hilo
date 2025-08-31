import json
from fastapi.testclient import TestClient
from backend.app import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_new_round_and_pending_flow():
    r = client.post(
        "/new-round",
        json={
            "players": [{"kind": "H", "name": "You"}, {"kind": "AI", "name": "Bot A"}],
            "aiDepth": 2,
        },
    )
    assert r.status_code == 200
    data = r.json()
    sid = data["sessionId"]
    state = data["state"]
    assert state["schemaVersion"] == 1
    assert "players" in state and len(state["players"]) == 2
    assert "pending" in state and isinstance(state["pending"], list)
    # resolve the first pending with a dummy response shaped by its kind
    if state["pending"]:
        a = state["pending"][0]
        aid = a["id"]
        kind = a["kind"]
        if kind == "choose_pos":
            resp = {"row": 0, "col": 0}
        elif kind == "reveal_card":
            resp = {"c": "R", "v": 0}
        else:
            resp = {"choice": "vertical"} if kind == "choose_diag_compact" else {"choice": "draw"}
        r2 = client.post("/resolve", json={"sessionId": sid, "actionId": aid, "response": resp})
        assert r2.status_code == 200
        state2 = r2.json()["state"]
        assert "pending" in state2


def test_step_endpoint_progresses():
    r = client.post(
        "/new-round",
        json={"players": [{"kind": "H", "name": "You"}, {"kind": "AI", "name": "Bot A"}]},
    )
    assert r.status_code == 200
    sid = r.json()["sessionId"]
    r2 = client.post("/step", json={"sessionId": sid})
    assert r2.status_code == 200
    state = r2.json()["state"]
    assert "players" in state

