# HILO — Console Arbiter (Physical-Deck Assistant)

A minimal, **console-based arbiter** for the card game **HILO** (Gigamic/Schmidt).  
This tool is designed to **participate in a real game with a physical deck** — no in‑game shuffling or RNG.  
It manages turn flow, validates inputs, resolves HILOs, scores rounds, and provides an **AI opponent** with probabilistic decision‑making and deterministic Monte Carlo lookahead.

> Not affiliated with Gigamic or Schmidt. This project is an independent hobby implementation that follows the public rules of the game.
> For more info on the game : https://www.gigamic.com/jeux-de-voyage/770-hilo-3421272837311.html

---

## ✨ Features

- **Physical‑deck workflow**: every card identity is entered by the user (or taken from the discard). No randomization inside the program.
- **2–4 players** (humans and/or AI), configured via constants.
- **AI with card counting & EV**: tracks remaining cards, computes probabilities/expected values, and chooses actions that minimize its grid sum.
- **Deterministic Monte Carlo lookahead** (depth 1–2): evaluates consequences of placements and HILO chains using cloned state; reproducible.
- **Explain mode** (optional): prints the **top‑K candidate moves** with probabilities and EV, plus a one‑line reason for the final pick.
- **Faithful HILO rules** including diagonal compaction and chain resolution.
- **Correct scoring**: end‑trigger double applies unless the trigger has the **unique** strictly‑lowest total (tie on lowest → still doubles).
- **Clear logging**: standardized lines such as `AI_ACTION`, `AI_REPLACED`, `AI_DISCARDED`, `AI_FLIPPED`, `AI_EVAL`, etc.
- **Color legend** using the 8 letters (with Lime):  
  `R=Red, B=Blue, G=Green, Y=Yellow, P=Purple, O=Orange, L=Lime, M=Magenta`

---

## 🧩 Quick overview of rules handled

- Each player has a **3×3 grid**. At setup, each player **reveals two** positions (row/col 0..2) and inputs their **color+value**.
- After all players have revealed, **one card is revealed** and placed on the **discard** to start the round.
- **Starting player** is the one with the **highest sum** of their two revealed cards (ties by table order).
- On your turn: **take discard** (must place) **or draw** (place or discard; if you discard a drawn card, you **must flip** one hidden cell).
- **HILO**: 3 visible cards of the **same color** aligned in row/column/diagonal → remove those 3; place them on the discard with **smallest value on top**.  
  If diagonal, **compact** immediately either **vertically** (two rows of three) or **horizontally** (two columns of three), then re‑scan for chains.
- **End of round**: when a player has no hidden cards (or no cards), others take **one last turn**. Reveal all, resolve HILOs, sum values.  
  The **trigger** doubles **unless** they have the **unique** strictly‑lowest total (tie on lowest → still doubles).  
  Play rounds until someone reaches `SCORE_LIMIT` (default 10; set to 100 for full games); the lowest total wins (ties allowed).

---

## 🚀 Getting started

### Requirements
- Python **3.10+** (tested on Windows)
- A terminal

### Run
```bash
python hilo.py
```

Optional demo mode:
```bash
python hilo.py --demo
```

Note: `--demo` currently prints a banner to indicate demo mode but does not change the configured players or scoring by itself. Adjust the config at the top of `hilo.py` as needed (see below).

---

## ⚙️ Configuration (top of file)

```python
# Player slots (2–4 active). kind: "H" human, "AI" bot, "X" inactive.
PLAYERS = [
    {"kind": "H",  "name": "You"},
    {"kind": "AI", "name": "Bot A"},
    {"kind": "AI", "name": "Bot B"},
    {"kind": "X",  "name": ""},
]

# AI depth for deterministic lookahead (known cards only).
AI_DEPTH: Literal[1, 2] = 2

# Optional explain mode
PRINT_PROBA: bool = True        # show EV/probability analysis
PRINT_TOP_K: int = 3            # show top-K placements
PRINT_PROBA_PREC: int = 3       # decimals for probabilities

# Scoring limit (end game when any player reaches this total)
SCORE_LIMIT = 10  # set to 100 for full-length games

# Color legend
COLOR_MAP = {
    "R": "Red", "B": "Blue", "G": "Green", "Y": "Yellow",
    "P": "Purple", "O": "Orange", "L": "Lime", "M": "Magenta",
}
```

> **Hints:**
> - Set `"X"` for unused player slots. Only non‑`"X"` entries are used to build the table (2–4 players).
> - For a full game, change `SCORE_LIMIT` to `100` (the current default is `10` for quick demos/tests).

---

## 🧠 AI: how it decides

- Maintains a **DeckTracker** per round (8×13 cards). It removes cards as they are revealed/replaced/removed so it can compute:
  - value and color distributions for **remaining unseen** cards,
  - probabilities like `P(V_hidden > v)` and expectations like `E[V_hidden]`.
- **Visible vs hidden replacement**: when holding a concrete card `D`, the AI evaluates replacing a **known** value (deterministic delta) vs replacing a **hidden** cell using the **expected value of the unseen distribution**.  
  Example: with `D = -1` and only a known `0` while most cells are unknown, **replacing hidden** is often better than replacing `0`.
- **HILO awareness**: immediate (deterministic) HILOs are applied during evaluation; color‑likelihoods encourage aligning colors toward HILO.
- **Deterministic Monte Carlo** (depth 1–2): runs on **cloned state + tracker snapshots**; may use a fixed seed for reproducibility; never mutates real state.

When `PRINT_PROBA=True`, each AI turn shows:
```
AI_EVAL: take_discard -> EV=<score_after>
AI_EVAL: draw_place   -> EV=<score_after>
AI_EVAL: draw_discard_flip -> EV=<score_after>, p_flip_improve=<prob>

AI_EVAL_CANDIDATE: PLACE=(r,c), kind=hidden|visible, result=<result_score>, Δ=<delta>,
                   p_improve=<p>, ev_delta_hidden=<ev>, hilo=<bool>, removed_sum=<sum>

AI_EVAL_PICK: PLACE=(r,c) because min(result)=<best_score> ...
```
And after acting:
```
AI_ACTION: discard; PLACE: (r,c)
AI_REPLACED: Y9
# or
AI_ACTION: draw; DISCARD; FLIP: (r,c)
AI_DISCARDED: M8
AI_FLIPPED: (r,c) -> B10
```

---

## 🕹️ I/O flow (human & AI)

- **Startup**: for each player, you input two reveals (positions + color+value). Then the program asks for the **initial discard top**.
- **Human turn**: the program waits for your real move, then prompts for exact details (take‑discard/draw, drawn card if any, replace or discard+flip; if diagonal HILO → choose vertical/horizontal).
- **AI turn**: the AI announces its intended move. If it draws, the program asks **you** to input the **actual drawn card** from the physical deck, then proceeds.



