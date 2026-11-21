"""Exact 3x3x3 cubie-level Rubik's cube state, scramble generator, Kociemba solver & verifier.

Every move rotates the exact 9 cubies of that layer by +/-90 deg around the outward face normal.
Nothing is approximated: every visual state on screen is read directly from this simulation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import kociemba
import numpy as np

# Standard Western color scheme:
#   U (+Z) = white, D (-Z) = yellow, F (-Y) = green, B (+Y) = blue, R (+X) = red, L (-X) = orange
FACE_NORMAL: Dict[str, np.ndarray] = {
    "U": np.array([0, 0, 1], dtype=int),
    "R": np.array([1, 0, 0], dtype=int),
    "F": np.array([0, -1, 0], dtype=int),
    "D": np.array([0, 0, -1], dtype=int),
    "L": np.array([-1, 0, 0], dtype=int),
    "B": np.array([0, 1, 0], dtype=int),
}

FACE_COLOR: Dict[str, str] = {
    "U": "white",
    "R": "red",
    "F": "green",
    "D": "yellow",
    "L": "orange",
    "B": "blue",
}

# Looking directly at each face from outside: (right_dir, down_dir) matching Kociemba URFDLB net
FACE_BASIS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "U": (np.array([1, 0, 0]), np.array([0, -1, 0])),
    "R": (np.array([0, 1, 0]), np.array([0, 0, -1])),
    "F": (np.array([1, 0, 0]), np.array([0, 0, -1])),
    "D": (np.array([1, 0, 0]), np.array([0, 1, 0])),
    "L": (np.array([0, -1, 0]), np.array([0, 0, -1])),
    "B": (np.array([-1, 0, 0]), np.array([0, 0, -1])),
}

KOCIEMBA_FACES = ("U", "R", "F", "D", "L", "B")
OPPOSITE_FACE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}


def _exact_rot90(axis: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Exact integer 3x3 Rodrigues rotation for `quarter_turns` * (+90 deg) about integer unit `axis`."""
    q = quarter_turns % 4
    I = np.eye(3, dtype=int)
    if q == 0:
        return I
    ax = np.asarray(axis, dtype=int)
    K = np.array(
        [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]],
        dtype=int,
    )
    K2 = K @ K
    if q == 1:
        return I + K + K2
    if q == 2:
        return I + 2 * K2
    return I - K + K2


def parse_move(token: str) -> Tuple[str, int]:
    """Parse Singmaster token ('R', \"U'\", 'F2') -> (face, cw_quarter_turns in {1, -1, 2})."""
    token = token.strip()
    face = token[0]
    if face not in FACE_NORMAL:
        raise ValueError(f"Unknown face in move {token!r}")
    suf = token[1:]
    if suf == "":
        return face, 1
    if suf == "'":
        return face, -1
    if suf == "2":
        return face, 2
    raise ValueError(f"Unknown modifier in move {token!r}")


@dataclass(frozen=True)
class VerifiedSolve:
    scramble: List[str]
    solution: List[str]
    scramble_facelets: str
    qtm_count: int
    htm_count: int


class CubeState:
    """Exact 26-cubie state representation (excluding invisible center core (0,0,0))."""

    def __init__(self) -> None:
        self.homes: List[Tuple[int, int, int]] = [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)
        ]
        self.pos: List[np.ndarray] = [np.array(h, dtype=int) for h in self.homes]
        self.rot: List[np.ndarray] = [np.eye(3, dtype=int) for _ in self.homes]

    def copy(self) -> "CubeState":
        c = CubeState()
        c.pos = [p.copy() for p in self.pos]
        c.rot = [r.copy() for r in self.rot]
        return c

    def layer_indices(self, face: str) -> List[int]:
        n = FACE_NORMAL[face]
        return [i for i, p in enumerate(self.pos) if int(np.dot(p, n)) == 1]

    def apply_quarter(self, face: str, cw_sign: int) -> None:
        """Apply a single clockwise (cw_sign=+1) or counter-clockwise (cw_sign=-1) 90-deg turn."""
        n = FACE_NORMAL[face]
        # Clockwise looking at face (+n) is -90 deg right-hand rotation about +n
        R = _exact_rot90(n, -cw_sign)
        for i in self.layer_indices(face):
            self.pos[i] = R @ self.pos[i]
            self.rot[i] = R @ self.rot[i]

    def apply_move(self, token: str) -> None:
        face, turns = parse_move(token)
        if turns == 2:
            self.apply_quarter(face, 1)
            self.apply_quarter(face, 1)
        else:
            self.apply_quarter(face, turns)

    def apply_sequence(self, moves: List[str]) -> None:
        for m in moves:
            self.apply_move(m)

    def to_kociemba_string(self) -> str:
        """Return the 54-char URFDLB facelet string representing the current cube state."""
        # Map current integer position -> cubie index
        grid = {tuple(int(v) for v in p): idx for idx, p in enumerate(self.pos)}
        # Map home normal -> face letter
        norm_to_face = {tuple(int(v) for v in n): f for f, n in FACE_NORMAL.items()}
        out: List[str] = []
        for f in KOCIEMBA_FACES:
            n = FACE_NORMAL[f]
            right, down = FACE_BASIS[f]
            for row in (-1, 0, 1):
                for col in (-1, 0, 1):
                    p = n + col * right + row * down
                    idx = grid[tuple(int(v) for v in p)]
                    # Which home face of this cubie currently points along `n`?
                    # Current orientation maps home vector `h_n` to `rot[idx] @ h_n = n` => `h_n = rot[idx].T @ n`
                    home_n = self.rot[idx].T @ n
                    out.append(norm_to_face[tuple(int(v) for v in home_n)])
        return "".join(out)

    def is_solved(self) -> bool:
        """True iff every cubie is at its home position AND every outward sticker matches its face."""
        for i, home in enumerate(self.homes):
            if tuple(int(v) for v in self.pos[i]) != home:
                return False
            # Center cubies can rotate in-place by multiples of 90 deg; check sticker colors match
            if np.count_nonzero(home) > 1:
                if not np.array_equal(self.rot[i], np.eye(3, dtype=int)):
                    return False
            else:
                n = np.array(home, dtype=int)
                if not np.array_equal(self.rot[i] @ n, n):
                    return False
        return self.to_kociemba_string() == "".join(f * 9 for f in KOCIEMBA_FACES)


def generate_scramble(length: int = 21, seed: int = 202604) -> List[str]:
    """Generate a legal WCA-style random scramble with no redundant consecutive/commuting moves."""
    rng = random.Random(seed)
    faces = list(KOCIEMBA_FACES)
    mods = ("", "'", "2")
    seq: List[str] = []
    prev_face = ""
    prev2_face = ""
    while len(seq) < length:
        f = rng.choice(faces)
        if f == prev_face:
            continue
        if prev2_face == f and OPPOSITE_FACE[prev_face] == f:
            continue
        seq.append(f + rng.choice(mods))
        prev2_face = prev_face
        prev_face = f
    return seq


def build_verified_solve(seed: int = 202604, scramble_len: int = 21) -> VerifiedSolve:
    """Create a random legal scramble, solve via Kociemba, and verify every single move."""
    scramble = generate_scramble(length=scramble_len, seed=seed)
    c = CubeState()
    assert c.is_solved(), "Initial cube must be solved"
    c.apply_sequence(scramble)
    assert not c.is_solved(), "Scramble unexpectedly resulted in solved cube"
    facelets = c.to_kociemba_string()
    sol_str = kociemba.solve(facelets)
    solution = sol_str.split()

    # Verify step-by-step on a fresh copy
    c_verify = CubeState()
    c_verify.apply_sequence(scramble)
    assert c_verify.to_kociemba_string() == facelets
    for m in solution:
        c_verify.apply_move(m)
    assert c_verify.is_solved(), f"Solution failed verification! Final state: {c_verify.to_kociemba_string()}"

    qtm = sum(2 if m.endswith("2") else 1 for m in solution)
    return VerifiedSolve(
        scramble=scramble,
        solution=solution,
        scramble_facelets=facelets,
        qtm_count=qtm,
        htm_count=len(solution),
    )


def default_solve() -> VerifiedSolve:
    """Canonical verified solve used by both the animation and sound synthesis."""
    # Seed chosen so solution has a rich mix of R/U/L/D/F/B finger tricks & double flicks (~25-28 QTM)
    return build_verified_solve(seed=202607, scramble_len=21)


if __name__ == "__main__":
    sv = default_solve()
    print(f"Scramble ({len(sv.scramble)} HTM): {' '.join(sv.scramble)}")
    print(f"Facelets: {sv.scramble_facelets}")
    print(f"Solution ({sv.htm_count} HTM / {sv.qtm_count} QTM): {' '.join(sv.solution)}")
    print("Verification: PASSED (100% solved)")
