# Tetris Autoplayer

An automatic Tetris player written in Python. The AI evaluates every legal placement for the current piece, looks another piece ahead, and chooses the move that best balances line clears, stack height, holes, and surface bumpiness.

## Features

- Heuristic search with optimized lookahead.
- Bomb and discard actions when the AI detects an unsafe placement.
- Optional genetic training for the linear evaluation weights.

## Requirements

- Python 3.9 or newer.
- Tkinter for the default visualizer.
- Pygame for the Pygame visualizer.

Install the optional Pygame dependency with:

```bash
python3 -m pip install pygame
```

## Running the game

Run the Tkinter visualizer with the AI:

```bash
python3 visual.py
```

Run the same visualizer in manual mode with `--manual` (or `-m`):

```bash
python3 visual.py --manual
```

The Pygame frontend is also available:

```bash
python3 visual-pygame.py
python3 visual-pygame.py --manual
```

For a terminal-only view, use the curses frontend:

```bash
python3 cmdline.py
python3 cmdline.py --manual
```

The AI runs automatically by default. In manual mode, use the arrow keys to move and rotate pieces, Space to drop, `b` to use a bomb, `d` to discard a piece, and `q` or Escape to quit.

## Training

`train_lin.py` uses a genetic algorithm to search for better heuristic
weights. Training is intentionally separate from normal gameplay and can use all available CPU cores:

```bash
python3 train_lin.py
```

The script writes intermediate and final weights to JSON files. Copy a set of weights into `player.py` if you want to evaluate a newly trained strategy.

## Project structure

- `board.py` contains the game rules, piece movement, collisions, and scoring.
- `player.py` contains the AI and its board-evaluation heuristics.
- `adversary.py` generates reproducible random pieces.
- `visual.py`, `visual-pygame.py`, and `cmdline.py` provide frontends.
- `train_lin.py` evolves the heuristic weights.
- `constants.py` and `exceptions.py` contain shared configuration and errors.

The project is self-contained and does not depend on an external game server
or external service.

Note: The Tetris game itself was not developed or coded by me, only the autoplayer and algorithms were.