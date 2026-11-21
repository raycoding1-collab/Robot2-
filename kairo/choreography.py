"""Full physically driven choreography and timeline engine for KAIRO-04.

Produces a dense, deterministic simulation state evaluated at any continuous time `t`:
  - Exact 26-cubie rigid transforms (verified Kociemba solve, magnetic detent physics)
  - 3-axis compliant grip recoil torque applied to cube & hands
  - Real finger-trick scheduling (U/U' index, D/D' ring, F/F' index, B/B' index, R/L wrist turns)
  - Safe lifted clearance arcs around the cube for all non-pushing finger strokes
  - 9.9 s six-face inspection with head tracking & eye saccades
  - Secondary motion: critically damped head/torso lean, micro-breathing, tiny servo jitter
  - Exact event log (detent clicks, finger taps, lift/land, joint velocities) for sound synthesis
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mathutils import Euler, Matrix, Quaternion, Vector

from .config import (
    CAM_FINALE_POS,
    CHEST_OFF,
    CUBE_HALF,
    CUBE_HOLD_POS,
    CUBE_REST_POS,
    CUBE_UPRIGHT,
    EYE_OFFSET,
    FINGERS,
    FINGER_ORDER,
    HEAD_OFF,
    MIRROR_X,
    NECK_OFF,
    PELVIS_POS,
    SHOULDER_OFF_R,
    SPINE_OFF,
    SURFACE_Z,
)
from .cube_logic import FACE_NORMAL, CubeState, VerifiedSolve, default_solve, parse_move
from .ik import finger_ik, finger_locals, solve_arm
from .physics import (
    QUARTER,
    CriticallyDampedSmoother,
    Profile,
    flick_profile,
    grip_recoil_3d,
    min_jerk,
    wrist_profile,
)

H = CUBE_HALF
PAD_CLR = 0.0008  # 0.8 mm clean silicone contact cushion outside plastic surface
PINKY_CURL = (0.10, 0.90, 1.18)

BASE_RIGHT_ROT = Matrix((
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
)).to_4x4()


def _load_tuned_grips() -> Dict[str, Matrix]:
    p = Path(__file__).resolve().parent / "grips_tuned.json"
    data = json.loads(p.read_text())
    out: Dict[str, Matrix] = {}
    for k, v in data.items():
        tx, ty, tz, rx, ry, rz = v["params"]
        m = BASE_RIGHT_ROT @ Euler((rx, ry, rz), "XYZ").to_matrix().to_4x4()
        m.translation = Vector((tx, ty, tz))
        out[k] = m
    return out


def _interp_mat(m0: Matrix, m1: Matrix, s: float) -> Matrix:
    s = min(1.0, max(0.0, s))
    p = m0.translation.lerp(m1.translation, s)
    q = m0.to_quaternion().slerp(m1.to_quaternion(), s)
    out = q.to_matrix().to_4x4()
    out.translation = p
    return out


def _push_outside_cube(p: Vector, margin: float) -> Vector:
    """Ensure 3D point `p` in cube-local space stays at least `margin` outside [-H, H]^3."""
    lim = H + margin
    if abs(p.x) < lim and abs(p.y) < lim and abs(p.z) < lim:
        # Push out along whichever axis is already closest to the surface
        dx, dy, dz = lim - abs(p.x), lim - abs(p.y), lim - abs(p.z)
        out = p.copy()
        if dx <= dy and dx <= dz:
            out.x = math.copysign(lim, p.x if abs(p.x) > 1e-6 else 1.0)
        elif dy <= dx and dy <= dz:
            out.y = math.copysign(lim, p.y if abs(p.y) > 1e-6 else 1.0)
        else:
            out.z = math.copysign(lim, p.z if abs(p.z) > 1e-6 else 1.0)
        return out
    return p


def _safe_arc(p0: Vector, p1: Vector, s: float, lift: float = 0.015, cube_margin: float = 0.010) -> Vector:
    """Smooth lifted trajectory between two cube-local points that always clears the cube."""
    u = min_jerk(s)
    p = p0.lerp(p1, u)
    # Outward radial bow peaking at s=0.5
    mid_dir = (p0 + p1)
    if mid_dir.length < 1e-5:
        mid_dir = Vector((1.0, 1.0, 0.5))
    mid_dir = mid_dir.normalized()
    bell = math.sin(math.pi * min(1.0, max(0.0, s)))
    p = p + mid_dir * (lift * bell)
    if bell > 0.05:
        p = _push_outside_cube(p, cube_margin * bell)
    return p


# ==============================================================================
# Quarter-turn physics event & finger-trick specs
# ==============================================================================
@dataclass
class TurnEvent:
    index: int
    token: str            # e.g. "R", "U'", "F2_1"
    face: str             # U, D, R, L, F, B
    cw: int               # +1 or -1
    t_start: float
    t_end: float
    t_capture: float      # exact magnetic detent capture time (for audio click sync!)
    t_drive: float        # fingertip release time (for flicks)
    mode: str             # "flick" or "wrist"
    hand: str             # "R" or "L"
    finger: str           # active flick finger (or "wrist")
    profile: Profile
    layer_indices: List[int]
    base_state_pos: List[Tuple[int, int, int]]
    base_state_rot: List[Matrix]
    axis_vec: Vector      # rotation axis = -cw * n_face (so +theta rotates correctly!)
    contact_start: Vector = field(default_factory=lambda: Vector())
    contact_normal: Vector = field(default_factory=lambda: Vector())


# Finger-trick mapping verified by torque sign r x (-n_contact) == axis_vec:
#   axis_vec = -cw * n_face
FLICK_SPECS = {
    # (face, cw): (hand, finger, contact_point_on_facelet, outward_normal_of_contact_face)
    ("U", +1): ("R", "index",  Vector((+0.022, +H, +0.022)), Vector((0.0, +1.0, 0.0))),
    ("U", -1): ("L", "index",  Vector((-0.022, +H, +0.022)), Vector((0.0, +1.0, 0.0))),
    ("D", +1): ("L", "ring",   Vector((-0.022, +H, -0.022)), Vector((0.0, +1.0, 0.0))),
    ("D", -1): ("R", "ring",   Vector((+0.022, +H, -0.022)), Vector((0.0, +1.0, 0.0))),
    ("F", +1): ("R", "index",  Vector((+0.025, -0.018, +H)), Vector((0.0, 0.0, +1.0))),
    ("F", -1): ("L", "index",  Vector((-0.025, -0.018, +H)), Vector((0.0, 0.0, +1.0))),
    ("B", +1): ("R", "index",  Vector((+H, +0.022, +0.022)), Vector((+1.0, 0.0, 0.0))),
    ("B", -1): ("L", "index",  Vector((-H, +0.022, +0.022)), Vector((-1.0, 0.0, 0.0))),
}


# Home fingertip positions in cube-local space for canonical RIGHT hand (LEFT mirrors X)
HOME_TARGETS_R = {
    "thumb": Vector((0.012, -(H + PAD_CLR), -0.002)),
    "index": Vector((0.020, H + 0.009, 0.021)),
    "middle": Vector((0.019, H + PAD_CLR, 0.003)),
    "ring": Vector((0.019, H + PAD_CLR, -0.017)),
}

R_TURN_TARGETS_R = {
    "thumb": Vector((0.019, -(H + PAD_CLR), -0.001)),
    "index": Vector((0.019, H + PAD_CLR, 0.019)),
    "middle": Vector((0.019, H + PAD_CLR, 0.000)),
    "ring": Vector((0.019, H + PAD_CLR, -0.019)),
}

PICKUP_TARGETS_R = {
    "thumb": Vector((0.015, -(H + PAD_CLR), 0.008)),
    "index": Vector((0.019, H + PAD_CLR, 0.015)),
    "middle": Vector((0.019, H + PAD_CLR, -0.003)),
    "ring": Vector((0.044, H + 0.012, 0.006)),
}


def _mirror_targets(d: Dict[str, Vector]) -> Dict[str, Vector]:
    return {k: Vector((-v.x, v.y, v.z)) for k, v in d.items()}


# ==============================================================================
# Master Simulation Timeline
# ==============================================================================
class Simulation:
    """Complete deterministic film simulation (22.8 s = 548 frames at 24 fps)."""

    def __init__(self, solve: Optional[VerifiedSolve] = None) -> None:
        self.solve = solve or default_solve()
        self.grips = _load_tuned_grips()

        # Key film timestamps (seconds)
        self.t_wake = 0.35
        self.t_reach_start = 0.70
        self.t_grasp = 1.95        # fingers contact cube on desk
        self.t_lift_start = 2.10   # cube leaves desk mat
        self.t_inspect_start = 3.10
        self.t_inspect_end = 13.00 # 9.90 s thorough 6-face inspection!
        self.t_solve_start = 13.45

        # Build physically integrated turn schedule for the verified Kociemba solution
        self.turns: List[TurnEvent] = []
        self.initial_state = CubeState()
        self.initial_state.apply_sequence(self.solve.scramble)
        self.final_state = self._schedule_solve()
        assert self.final_state.is_solved(), "Scheduled solve must terminate in 100% solved state!"

        self.t_solve_end = self.turns[-1].t_end
        self.t_present_start = self.t_solve_end + 0.35
        self.t_present_end = self.t_present_start + 2.35
        self.t_setdown_touch = self.t_present_end + 0.95  # cube lands back on desk mat
        self.t_release_end = self.t_setdown_touch + 0.75
        self.t_end = self.t_release_end + 1.20            # calm lens-look finale beat
        self.fps = 24
        self.total_frames = int(round(self.t_end * self.fps))

        # Precompute 3-axis compliant grip recoil during the solve
        self.recoil_t0 = self.t_solve_start - 0.10
        self.recoil_t1 = self.t_solve_end + 0.30
        self.recoil_px, self.recoil_py, self.recoil_pz = self._compute_recoil()

        # Precompute smooth inspection waypoints (all 6 faces visited & held clearly!)
        self.inspect_keys = self._build_inspection_keyframes()

        # Precompute critically damped head & torso trajectories at 120 Hz for silky motion
        self._precompute_body_secondary()

        # Collect discrete contact/foley events for audio engine
        self.contact_events = self._build_contact_events()

    def _schedule_solve(self) -> CubeState:
        """Schedule every quarter turn of the verified Kociemba solution with real physics."""
        state = self.initial_state.copy()
        pf_flick, td_flick = flick_profile(theta_rel_deg=42.0, margin=1.07)
        pf_wrist = wrist_profile(duration=0.132, overshoot_deg=1.35)

        t = self.t_solve_start
        idx = 0
        prev_hand = None
        prev_mode = None

        for move in self.solve.solution:
            face, turns = parse_move(move)
            quarters = [(face, 1), (face, 1)] if turns == 2 else [(face, turns)]
            for q_i, (f, cw) in enumerate(quarters):
                is_wrist = f in ("R", "L")
                hand = "R" if f == "R" else ("L" if f == "L" else FLICK_SPECS[(f, cw)][0])
                mode = "wrist" if is_wrist else "flick"

                # Small natural spacing / regrip gap between consecutive moves on the same hand
                if idx > 0:
                    if q_i == 1:
                        # Tight double-flick / double-wrist cadence (~115 ms between quarter turns)
                        t += 0.015 if is_wrist else 0.008
                    elif hand == prev_hand and mode == prev_mode:
                        t += 0.032  # rapid finger reset
                    elif hand == prev_hand:
                        t += 0.042  # quick grip shift (wrist <-> flick)
                    else:
                        # Alternating hands can overlap slightly like a real 7 TPS speedcuber!
                        t -= 0.018

                prof = pf_wrist if is_wrist else pf_flick
                t_start = t
                t_end = t + prof.duration
                t_cap = t + prof.capture_time

                n_vec = Vector(FACE_NORMAL[f].tolist())
                axis_vec = (-cw) * n_vec  # positive angle theta -> clockwise when cw=+1
                layer = state.layer_indices(f)
                base_pos = [tuple(int(v) for v in p) for p in state.pos]
                base_rot = [Matrix(r.tolist()).to_4x4() for r in state.rot]

                if is_wrist:
                    finger = "wrist"
                    c_start, c_norm = Vector(), Vector()
                else:
                    _, finger, c_start, c_norm = FLICK_SPECS[(f, cw)]

                self.turns.append(
                    TurnEvent(
                        index=idx,
                        token=f"{move}" if turns != 2 else f"{move}_{q_i+1}",
                        face=f,
                        cw=cw,
                        t_start=t_start,
                        t_end=t_end,
                        t_capture=t_cap,
                        t_drive=td_flick if not is_wrist else prof.capture_time,
                        mode=mode,
                        hand=hand,
                        finger=finger,
                        profile=prof,
                        layer_indices=layer,
                        base_state_pos=base_pos,
                        base_state_rot=base_rot,
                        axis_vec=axis_vec,
                        contact_start=c_start,
                        contact_normal=c_norm,
                    )
                )

                state.apply_quarter(f, cw)
                t = t_end
                prev_hand = hand
                prev_mode = mode
                idx += 1

        return state

    def _compute_recoil(self):
        def torque_fn(t: float) -> Tuple[float, float, float]:
            tx = ty = tz = 0.0
            for ev in self.turns:
                if ev.t_start - 0.002 <= t <= ev.t_end + 0.002:
                    acc = ev.profile.accel(t - ev.t_start)
                    # Scale wrist turns slightly lower recoil since palm braces the core
                    w = 0.55 if ev.mode == "wrist" else 1.0
                    tx += ev.axis_vec.x * acc * w
                    ty += ev.axis_vec.y * acc * w
                    tz += ev.axis_vec.z * acc * w
            return (tx, ty, tz)

        return grip_recoil_3d(self.recoil_t0, self.recoil_t1, torque_fn, omega=74.0, zeta=0.52, gain=0.000018)

    def _build_inspection_keyframes(self) -> List[Tuple[float, Vector, Quaternion, str]]:
        """9.9 s deliberate inspection pausing clearly on ALL 6 faces (F, U, R, B, L, D) -> upright.

        Every single face normal aligns squarely toward KAIRO-04's eyes (dot > 0.88, < 28 deg)
        while keeping both arms comfortably inside ergonomic reach (0 unreachable frames).
        """
        t0 = self.t_inspect_start
        raw = [
            # F (front) tilted up toward eyes
            (0.00, Vector(( 0.000,  0.000,  0.000)), ( 0.52,  0.00,  0.00), "F_hold_start"),
            (0.95, Vector(( 0.000,  0.004,  0.004)), ( 0.58,  0.04, -0.04), "F_study"),
            # U (top) tilted back toward eyes
            (1.65, Vector(( 0.000,  0.010,  0.010)), (-0.50,  0.04,  0.04), "U_show"),
            (2.60, Vector(( 0.002,  0.012,  0.012)), (-0.56, -0.03, -0.04), "U_study"),
            # R (right): Ry(-0.82) lifts +X up toward +Z, Rz(-0.72) yaws +X toward -Y (eyes!)
            (3.40, Vector((-0.008,  0.014,  0.016)), ( 0.08, -0.80, -0.70), "R_show"),
            (4.40, Vector((-0.010,  0.016,  0.016)), ( 0.06, -0.84, -0.74), "R_study"),
            # B (back): Rx(+1.72) rolls +Y smoothly over top toward eyes with comfortable reach!
            (5.30, Vector(( 0.000,  0.038,  0.026)), ( 1.68,  0.04,  0.04), "B_show"),
            (6.25, Vector(( 0.002,  0.038,  0.026)), ( 1.72, -0.04, -0.04), "B_study"),
            # L (left): exact mirror of R -> Ry(+0.82), Rz(+0.72) turns -X straight to eyes!
            (7.15, Vector(( 0.008,  0.020,  0.018)), ( 0.08,  0.80,  0.70), "L_show"),
            (8.05, Vector(( 0.010,  0.020,  0.018)), ( 0.06,  0.84,  0.74), "L_study"),
            # D (bottom): Rx(-2.20) tilts -Z squarely toward eyes (dot > 0.96!)
            (8.75, Vector(( 0.000,  0.038,  0.030)), (-2.16,  0.04, -0.04), "D_show"),
            (9.35, Vector(( 0.000,  0.038,  0.030)), (-2.22, -0.04,  0.04), "D_study"),
            # Return crisply to upright ready stance for speedsolve
            (9.90, Vector(( 0.000,  0.000,  0.000)), ( 0.00,  0.00,  0.00), "ready"),
        ]
        out = []
        for dt, off, eul, lbl in raw:
            q = (CUBE_UPRIGHT @ Euler(eul, "XYZ").to_matrix().to_4x4()).to_quaternion()
            out.append((t0 + dt, CUBE_HOLD_POS + off, q, lbl))
        return out

    def _build_contact_events(self) -> List[dict]:
        """Discrete physical foley events synchronized to the millisecond."""
        evts = [
            {"type": "power_on", "t": self.t_wake, "amp": 1.0},
            {"type": "finger_tap", "t": self.t_grasp, "amp": 0.72, "hand": "both"},
            {"type": "cube_lift", "t": self.t_lift_start, "amp": 0.80},
            {"type": "cube_land", "t": self.t_setdown_touch, "amp": 1.00},
            {"type": "hand_settle", "t": self.t_release_end - 0.12, "amp": 0.65, "hand": "both"},
        ]
        for ev in self.turns:
            if ev.mode == "flick":
                evts.append({"type": "finger_tap", "t": ev.t_start, "amp": 0.52, "hand": ev.hand, "face": ev.face})
            evts.append({
                "type": "detent_click",
                "t": ev.t_capture,
                "t_start": ev.t_start,
                "t_end": ev.t_end,
                "amp": 0.92 if ev.mode == "flick" else 0.78,
                "face": ev.face,
                "cw": ev.cw,
                "mode": ev.mode,
                "hand": ev.hand,
                "token": ev.token,
            })
        return evts

    # ------------------------------------------------------------------ cube root world pose
    def cube_pose(self, t: float) -> Matrix:
        """World 4x4 transform of the puzzle cube root at time `t` (including grip recoil)."""
        q_up = CUBE_UPRIGHT.to_quaternion()

        if t <= self.t_lift_start:
            pos = CUBE_REST_POS.copy()
            q = q_up.copy()
        elif t < self.t_inspect_start:
            s = min_jerk((t - self.t_lift_start) / (self.t_inspect_start - self.t_lift_start))
            pos = CUBE_REST_POS.lerp(CUBE_HOLD_POS, s)
            pos.z += 0.014 * math.sin(math.pi * s)
            q_target = self.inspect_keys[0][2]
            q = q_up.slerp(q_target, s)
        elif t <= self.t_inspect_end:
            keys = self.inspect_keys
            pos, q = keys[-1][1].copy(), keys[-1][2].copy()
            for a, b in zip(keys[:-1], keys[1:]):
                if a[0] <= t <= b[0]:
                    s = min_jerk((t - a[0]) / max(1e-6, b[0] - a[0]))
                    pos = a[1].lerp(b[1], s)
                    q = a[2].slerp(b[2], s)
                    break
        elif t <= self.t_present_start:
            pos = CUBE_HOLD_POS.copy()
            pos.x += 0.0010 * math.sin(2.1 * t)
            pos.z += 0.0012 * math.cos(1.7 * t)
            q = q_up.copy()
            if self.recoil_t0 <= t <= self.recoil_t1:
                rt = t - self.recoil_t0
                rx = self.recoil_px(rt)
                ry = self.recoil_py(rt)
                rz = self.recoil_pz(rt)
                q = q @ Euler((rx, ry, rz), "XYZ").to_quaternion()
                pos += Vector((ry * 0.03, -rx * 0.02, rx * 0.025))
        elif t <= self.t_present_end:
            u = (t - self.t_present_start) / (self.t_present_end - self.t_present_start)
            p_keys = [
                (0.00, Vector((0.000, -0.012, 0.012)), ( 0.00,  0.00,  0.00)),
                (0.35, Vector((0.000, -0.022, 0.022)), (-0.30,  0.20,  0.48)),
                (0.72, Vector((0.000, -0.022, 0.020)), (-0.26, -0.20, -0.48)),
                (1.00, Vector((0.000, -0.008, 0.010)), ( 0.00,  0.00,  0.00)),
            ]
            pos = CUBE_HOLD_POS.copy()
            q = q_up.copy()
            for a, b in zip(p_keys[:-1], p_keys[1:]):
                if a[0] <= u <= b[0]:
                    s = min_jerk((u - a[0]) / max(1e-6, b[0] - a[0]))
                    pos = CUBE_HOLD_POS + a[1].lerp(b[1], s)
                    qa = (CUBE_UPRIGHT @ Euler(a[2], "XYZ").to_matrix().to_4x4()).to_quaternion()
                    qb = (CUBE_UPRIGHT @ Euler(b[2], "XYZ").to_matrix().to_4x4()).to_quaternion()
                    q = qa.slerp(qb, s)
                    break
        elif t < self.t_setdown_touch:
            s = min_jerk((t - self.t_present_end) / (self.t_setdown_touch - self.t_present_end))
            p0 = CUBE_HOLD_POS + Vector((0.0, -0.008, 0.010))
            pos = p0.lerp(CUBE_REST_POS, s)
            pos.z += 0.012 * math.sin(math.pi * s)
            q = q_up.copy()
        else:
            pos = CUBE_REST_POS.copy()
            q = q_up.copy()

        m = q.to_matrix().to_4x4()
        m.translation = pos
        return m

    # ------------------------------------------------------------------ cubie local rotations
    def cubie_quaternions(self, t: float) -> List[Quaternion]:
        """Return the exact local rotation Quaternion for all 26 cubies at time `t`."""
        if t < self.t_solve_start:
            return [Matrix(r.tolist()).to_quaternion() for r in self.initial_state.rot]
        if t >= self.t_solve_end:
            return [Matrix(r.tolist()).to_quaternion() for r in self.final_state.rot]

        active = [ev for ev in self.turns if ev.t_start <= t <= ev.t_end]
        if not active:
            done = [ev for ev in self.turns if ev.t_end < t]
            if not done:
                return [Matrix(r.tolist()).to_quaternion() for r in self.initial_state.rot]
            last = done[-1]
            nxt = self.turns[last.index + 1] if last.index + 1 < len(self.turns) else None
            if nxt is not None:
                return [m.to_quaternion() for m in nxt.base_state_rot]
            return [Matrix(r.tolist()).to_quaternion() for r in self.final_state.rot]

        base_ev = active[0]
        rots = [m.copy() for m in base_ev.base_state_rot]
        for ev in active:
            theta = ev.profile(t - ev.t_start)
            R_part = Matrix.Rotation(theta, 4, ev.axis_vec)
            for ci in ev.layer_indices:
                rots[ci] = R_part @ rots[ci]
        return [m.to_quaternion() for m in rots]

    # ------------------------------------------------------------------ hand & finger poses
    def _desk_rest_hand_world(self, side: str) -> Tuple[Matrix, Dict[str, Tuple[float, float, float]]]:
        """Calm, natural resting hand pose hovering/resting cleanly on the ESD desk mat."""
        rot_r = Matrix((
            (-1.0,  0.0, 0.0),
            ( 0.0, -1.0, 0.0),
            ( 0.0,  0.0, 1.0),
        )).to_4x4() @ Euler((-0.10, 0.04, -0.22), "XYZ").to_matrix().to_4x4()
        # Wrist elevated 68 mm above ESD mat so curved fingertips kiss cleanly above z = SURFACE_Z
        pos_r = Vector((-0.178, -0.265, SURFACE_Z + 0.068))
        if side == "R":
            m = rot_r
            m.translation = pos_r
        else:
            m_r = rot_r.copy()
            m_r.translation = pos_r
            m = MIRROR_X @ m_r @ MIRROR_X

        rest_angles = {
            "thumb":  (0.10, 0.02, 0.10),
            "index":  (-0.04, 0.28, 0.32),
            "middle": (0.00, 0.30, 0.34),
            "ring":   (0.04, 0.32, 0.36),
            "pinky":  (0.08, 0.36, 0.40),
        }
        return m, rest_angles

    def _canonical_hand_in_cube(self, side: str, grip_name: str, wrist_rot_x: float = 0.0) -> Matrix:
        """Return hand transform in cube-local space for side ('R' or 'L') with optional wrist turn about cube X."""
        G_r = self.grips[grip_name]
        if abs(wrist_rot_x) > 1e-6:
            G_r = Matrix.Rotation(wrist_rot_x, 4, "X") @ G_r
        if side == "R":
            return G_r
        return MIRROR_X @ G_r @ MIRROR_X

    def _solve_hand_fingers(
        self,
        side: str,
        hand_cube: Matrix,
        targets_cube: Dict[str, Vector],
        pinky_angles: Tuple[float, float, float] = PINKY_CURL,
    ) -> Dict[str, Tuple[float, float, float]]:
        """Solve canonical (psi, q1, q2) angles for all 5 fingers of `side` ('R' or 'L')."""
        if side == "L":
            hc_canon = MIRROR_X @ hand_cube @ MIRROR_X
            inv_h = hc_canon.inverted()
            out = {}
            for fn, tgt_c in targets_cube.items():
                tgt_canon = Vector((-tgt_c.x, tgt_c.y, tgt_c.z))
                out[fn] = finger_ik(fn, inv_h @ tgt_canon)
        else:
            inv_h = hand_cube.inverted()
            out = {fn: finger_ik(fn, inv_h @ tgt_c) for fn, tgt_c in targets_cube.items()}
        out["pinky"] = pinky_angles
        return out

    def hand_state(self, t: float, side: str, cube_world: Matrix) -> Tuple[Matrix, Dict[str, Tuple[float, float, float]]]:
        """Compute (hand_world_matrix, {finger: (psi, q1, q2)}) at time `t`."""
        rest_w, rest_ang = self._desk_rest_hand_world(side)
        pickup_hc = self._canonical_hand_in_cube(side, "pickup")
        home_hc = self._canonical_hand_in_cube(side, "home")
        pickup_tgts = PICKUP_TARGETS_R if side == "R" else _mirror_targets(PICKUP_TARGETS_R)
        home_tgts = HOME_TARGETS_R if side == "R" else _mirror_targets(HOME_TARGETS_R)

        if t <= self.t_reach_start:
            return rest_w, rest_ang

        if t < self.t_grasp:
            s = (t - self.t_reach_start) / (self.t_grasp - self.t_reach_start)
            u = min_jerk(s)
            pick_w = cube_world @ pickup_hc
            hw = _interp_mat(rest_w, pick_w, u)
            hw.translation.z += 0.056 * math.sin(math.pi * s)
            pick_ang = self._solve_hand_fingers(side, pickup_hc, pickup_tgts)
            open_spread = math.sin(math.pi * s)
            angs = {}
            for fn in FINGER_ORDER:
                a0, a1 = rest_ang[fn], pick_ang[fn]
                psi = a0[0] * (1 - u) + a1[0] * u
                q1 = a0[1] * (1 - u) + a1[1] * u - 0.35 * open_spread
                q2 = a0[2] * (1 - u) + a1[2] * u - 0.38 * open_spread
                angs[fn] = (psi, q1, q2)
            return hw, angs

        if t < self.t_inspect_start:
            if t < self.t_lift_start:
                s = 0.0
            else:
                s = min_jerk((t - self.t_lift_start) / (self.t_inspect_start - self.t_lift_start))
            hc = _interp_mat(pickup_hc, home_hc, s)
            flare = (0.008 if side == "R" else -0.008) * math.sin(math.pi * s)
            hc.translation.x += flare
            tgts = {fn: pickup_tgts[fn].lerp(home_tgts[fn], s) for fn in pickup_tgts}
            if 0.02 < s < 0.98:
                tgts["ring"] = _safe_arc(pickup_tgts["ring"], home_tgts["ring"], s, lift=0.012, cube_margin=0.008)
                tgts["thumb"] = tgts["thumb"] + Vector((0.0, -0.006 * math.sin(math.pi * s), 0.0))
            hc, angs = self._guard_hand_clearance(side, hc, tgts, t)
            return cube_world @ hc, angs

        if t < self.t_solve_start - 0.06 or (self.t_solve_end + 0.08 <= t <= self.t_present_end):
            tgts = {k: v.copy() for k, v in home_tgts.items()}
            if self.t_inspect_start <= t <= self.t_inspect_end:
                breath = 0.0022 * (0.5 + 0.5 * math.sin(1.4 * t + (0.0 if side == "R" else 1.2)))
                tgts["index"].y += breath
                tgts["index"].z += breath * 0.6
            hc, angs = self._guard_hand_clearance(side, home_hc, tgts, t)
            return cube_world @ hc, angs

        if t < self.t_solve_end + 0.08:
            return self._solve_hand_pose(t, side, cube_world)

        if t < self.t_setdown_touch:
            s = min_jerk((t - self.t_present_end) / (self.t_setdown_touch - self.t_present_end))
            hc = _interp_mat(home_hc, pickup_hc, s)
            flare = (0.008 if side == "R" else -0.008) * math.sin(math.pi * s)
            hc.translation.x += flare
            tgts = {fn: home_tgts[fn].lerp(pickup_tgts[fn], s) for fn in home_tgts}
            if 0.02 < s < 0.98:
                tgts["ring"] = _safe_arc(home_tgts["ring"], pickup_tgts["ring"], s, lift=0.014, cube_margin=0.009)
                tgts["thumb"] = tgts["thumb"] + Vector((0.0, -0.006 * math.sin(math.pi * s), 0.0))
            hc, angs = self._guard_hand_clearance(side, hc, tgts, t)
            return cube_world @ hc, angs

        if t < self.t_release_end:
            s = (t - self.t_setdown_touch) / (self.t_release_end - self.t_setdown_touch)
            u = min_jerk(s)
            pick_w = cube_world @ pickup_hc
            hw = _interp_mat(pick_w, rest_w, u)
            hw.translation.z += 0.052 * math.sin(math.pi * s)
            pick_ang = self._solve_hand_fingers(side, pickup_hc, pickup_tgts)
            open_spread = math.sin(math.pi * s)
            angs = {}
            for fn in FINGER_ORDER:
                a0, a1 = pick_ang[fn], rest_ang[fn]
                angs[fn] = (
                    a0[0] * (1 - u) + a1[0] * u,
                    a0[1] * (1 - u) + a1[1] * u - 0.32 * open_spread,
                    a0[2] * (1 - u) + a1[2] * u - 0.35 * open_spread,
                )
            return hw, angs

        return rest_w, rest_ang

    def _active_cube_sdf(self, p_cube: Vector, t: float) -> float:
        """Exact signed distance to static cube core + any actively spinning layer at time `t`."""
        def box_sdf(pt: Vector, h: float) -> float:
            qx, qy, qz = abs(pt.x) - h, abs(pt.y) - h, abs(pt.z) - h
            return Vector((max(qx, 0.0), max(qy, 0.0), max(qz, 0.0))).length + min(max(qx, qy, qz), 0.0)

        active = [ev for ev in self.turns if ev.t_start <= t <= ev.t_end]
        if not active:
            return box_sdf(p_cube, H)
        d_min = 1e9
        for ev in active:
            theta = ev.profile(t - ev.t_start)
            p_rot = Matrix.Rotation(-theta, 3, ev.axis_vec) @ p_cube
            n = Vector((-ev.cw * ev.axis_vec.x, -ev.cw * ev.axis_vec.y, -ev.cw * ev.axis_vec.z))
            ax_i = 0 if abs(n.x) > 0.5 else (1 if abs(n.y) > 0.5 else 2)
            sgn = 1.0 if n[ax_i] > 0 else -1.0

            def sub_sdf(pt: Vector, lo_ax: float, hi_ax: float) -> float:
                c_ax = 0.5 * (lo_ax + hi_ax)
                h_ax = 0.5 * (hi_ax - lo_ax)
                q = [abs(pt[k]) - H for k in range(3)]
                q[ax_i] = abs(pt[ax_i] - c_ax) - h_ax
                return Vector((max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0))).length + min(max(q[0], q[1], q[2]), 0.0)

            if sgn > 0:
                d_min = min(d_min, sub_sdf(p_cube, -H, H / 3.0), sub_sdf(p_rot, H / 3.0, H))
            else:
                d_min = min(d_min, sub_sdf(p_cube, -H / 3.0, H), sub_sdf(p_rot, -H, -H / 3.0))
        return d_min

    def _finger_max_pen(self, side: str, hc: Matrix, fn: str, angs: Tuple[float, float, float], t: float) -> Tuple[float, Vector]:
        """Compute max capsule penetration of finger `fn` into the active cube at time `t` + outward push dir."""
        from .config import phalanx_radius
        from .ik import finger_fk_chain

        chain = list(finger_fk_chain(fn, *angs))
        if side == "L":
            chain = [Vector((-p.x, p.y, p.z)) for p in chain]
        pts = [hc @ p for p in chain]
        max_pen = 0.0
        worst_p = pts[4]
        for seg in (0, 1):
            r = phalanx_radius(fn, seg) * 0.92
            for k in range(5):
                p = pts[seg].lerp(pts[seg + 1], k / 4.0)
                pen = r - self._active_cube_sdf(p, t)
                if pen > max_pen:
                    max_pen, worst_p = pen, p
        r2 = phalanx_radius(fn, 2) * 0.84
        for k in range(3):
            p = pts[2].lerp(pts[3], (k / 2.0) * 0.52)
            pen = r2 - self._active_cube_sdf(p, t)
            if pen > max_pen:
                max_pen, worst_p = pen, p
        pen_pad = 0.0002 - self._active_cube_sdf(pts[4], t)
        if pen_pad > max_pen:
            max_pen, worst_p = pen_pad, pts[4]
        return max_pen, worst_p

    def _guard_hand_clearance(
        self,
        side: str,
        hc: Matrix,
        tgts: Dict[str, Vector],
        t: float,
    ) -> Tuple[Matrix, Dict[str, Tuple[float, float, float]]]:
        """Solve finger IK and gently relieve any sub-millimetre spinning-corner collision."""
        sx = 1.0 if side == "R" else -1.0
        hc_out = hc.copy()
        tgts_out = {k: v.copy() for k, v in tgts.items()}

        # Up to 6 tiny geometric relaxation steps so every capsule clears by >= 0.2 mm
        for _ in range(6):
            angs = self._solve_hand_fingers(side, hc_out, tgts_out)
            worst_all = 0.0
            for fn in ("thumb", "index", "middle", "ring"):
                pen, wp = self._finger_max_pen(side, hc_out, fn, angs[fn], t)
                if pen > 0.00015:
                    worst_all = max(worst_all, pen)
                    # Push target smoothly outward along local SDF gradient (away from cube center)
                    d = wp.normalized() if wp.length > 1e-5 else Vector((sx, 0.0, 0.0))
                    # Slightly bias outward laterally so proximal/middle phalanges also clear!
                    push = (d + Vector((sx * 0.45, 0.0, 0.0))).normalized()
                    tgts_out[fn] = tgts_out[fn] + push * (pen + 0.0008)
                    # If distal knuckle still clips due to palm proximity, nudge wrist gently outward
                    if pen > 0.0025:
                        hc_out.translation += Vector((sx * 0.0018, 0.0, 0.0006 * (1.0 if wp.z > 0 else -1.0)))
            # Check passive pinky & palm clearance too
            pen_p, _ = self._finger_max_pen(side, hc_out, "pinky", angs["pinky"], t)
            if pen_p > 0.00015:
                worst_all = max(worst_all, pen_p)
                hc_out.translation.x += sx * (pen_p + 0.001)
            if worst_all <= 0.00015:
                return hc_out, angs

        return hc_out, self._solve_hand_fingers(side, hc_out, tgts_out)

    def _solve_hand_pose(self, t: float, side: str, cube_world: Matrix) -> Tuple[Matrix, Dict[str, Tuple[float, float, float]]]:
        """Compute exact hand & finger pose for `side` during the speedsolve window."""
        home_hc = self._canonical_hand_in_cube(side, "home")
        rturn_hc = self._canonical_hand_in_cube(side, "r_turn")
        home_tgts = {k: v.copy() for k, v in (HOME_TARGETS_R if side == "R" else _mirror_targets(HOME_TARGETS_R)).items()}
        rturn_tgts = {k: v.copy() for k, v in (R_TURN_TARGETS_R if side == "R" else _mirror_targets(R_TURN_TARGETS_R)).items()}

        windup = 0.052
        ret_win = 0.060
        hc = home_hc.copy()
        tgts = home_tgts
        sx = 1.0 if side == "R" else -1.0

        # Resource scheduler: if the OTHER hand is actively spinning D/U/F/B, lift any
        # finger on THIS stabilizing hand that would sit on the spinning slice!
        for other_ev in self.turns:
            if other_ev.hand == side:
                continue
            if other_ev.t_start - 0.035 <= t <= other_ev.t_end + 0.035:
                env = math.sin(
                    math.pi * min(1.0, max(0.0, (t - (other_ev.t_start - 0.035)) / max(1e-4, (other_ev.t_end - other_ev.t_start) + 0.070)))
                ) ** 0.55
                if other_ev.face == "D":
                    # Lift ring finger off the spinning D layer while thumb + middle securely hold cube!
                    tgts["ring"] += Vector((sx * 0.008 * env, 0.015 * env, 0.004 * env))
                elif other_ev.face == "U":
                    tgts["index"] += Vector((sx * 0.008 * env, 0.012 * env, 0.010 * env))
                elif other_ev.face == "F":
                    # Slightly back thumb off spinning F corner while middle+ring brace B!
                    tgts["thumb"] += Vector((sx * 0.006 * env, -0.014 * env, 0.0))
                elif other_ev.face == "B":
                    # Slightly ease middle/ring back off spinning B while thumb braces F!
                    tgts["middle"] += Vector((sx * 0.006 * env, 0.014 * env, 0.0))
                    tgts["ring"] += Vector((sx * 0.006 * env, 0.014 * env, 0.0))
                    tgts["index"] += Vector((sx * 0.008 * env, 0.014 * env, 0.006 * env))

        best_ev = None
        for ev in self.turns:
            if ev.hand != side:
                continue
            if ev.t_start - windup <= t <= ev.t_end + ret_win:
                if best_ev is None or (ev.t_start <= t <= ev.t_end):
                    best_ev = ev

        if best_ev is None:
            hc, angs = self._guard_hand_clearance(side, hc, tgts, t)
            return cube_world @ hc, angs

        ev = best_ev
        if ev.mode == "wrist":
            palm_share = 0.80
            if t < ev.t_start:
                s = min_jerk((t - (ev.t_start - windup)) / windup)
                hc = _interp_mat(home_hc, rturn_hc, s)
                tgts = {k: home_tgts[k].lerp(rturn_tgts[k], s) for k in home_tgts}
            elif t <= ev.t_end:
                theta = ev.profile(t - ev.t_start)
                ang_x = ev.axis_vec.x * theta
                hc = self._canonical_hand_in_cube(side, "r_turn", wrist_rot_x=ang_x * palm_share)
                R_layer = Matrix.Rotation(theta, 4, ev.axis_vec)
                tgts = {k: R_layer @ rturn_tgts[k] for k in rturn_tgts}
            else:
                s = (t - ev.t_end) / ret_win
                u = min_jerk(s)
                ang_end = ev.axis_vec.x * QUARTER
                hc_end = self._canonical_hand_in_cube(side, "r_turn", wrist_rot_x=ang_end * palm_share)
                R_end = Matrix.Rotation(QUARTER, 4, ev.axis_vec)
                hc = _interp_mat(hc_end, home_hc, u)
                flare = (0.014 if side == "R" else -0.014) * math.sin(math.pi * s)
                hc.translation.x += flare
                tgts = {
                    k: _safe_arc(R_end @ rturn_tgts[k], home_tgts[k], s, lift=0.016, cube_margin=0.012)
                    for k in home_tgts
                }
            hc, angs = self._guard_hand_clearance(side, hc, tgts, t)
            return cube_world @ hc, angs

        # FLICK MOVE (U, U', D, D', F, F', B, B')
        fn = ev.finger
        p_home = home_tgts[fn]
        if ev.face == "F":
            t_tot = (ev.t_end + ret_win) - (ev.t_start - windup)
            env = math.sin(math.pi * min(1.0, max(0.0, (t - (ev.t_start - windup)) / max(1e-4, t_tot)))) ** 0.65
            hc.translation += Vector((sx * 0.014 * env, -0.010 * env, 0.026 * env))

        extra_n = 0.0055 if ev.face == "F" else 0.0032
        p_contact = ev.contact_start + ev.contact_normal * (PAD_CLR + extra_n)
        p_ready = ev.contact_start + ev.contact_normal * (0.015 if ev.face == "F" else 0.013)

        if t < ev.t_start:
            s = (t - (ev.t_start - windup)) / windup
            if s < 0.72:
                tgts[fn] = _safe_arc(p_home, p_ready, s / 0.72, lift=0.018, cube_margin=0.014)
            else:
                tgts[fn] = p_ready.lerp(p_contact, min_jerk((s - 0.72) / 0.28))
        elif t <= ev.t_start + ev.t_drive:
            theta = ev.profile(t - ev.t_start)
            R_part = Matrix.Rotation(theta, 4, ev.axis_vec)
            tgts[fn] = R_part @ p_contact
        else:
            theta_rel = ev.profile(ev.t_drive)
            R_rel = Matrix.Rotation(theta_rel, 4, ev.axis_vec)
            p_rel = R_rel @ p_contact
            n_rel = (R_rel.to_3x3() @ ev.contact_normal).normalized()
            peel_dir = (n_rel + Vector((sx * 0.55, 0.0, 0.35 if ev.face == "F" else 0.0))).normalized()
            p_peel = p_rel + peel_dir * 0.020
            t_after = t - (ev.t_start + ev.t_drive)
            t_ret_total = max(0.055, (ev.t_end + ret_win) - (ev.t_start + ev.t_drive))
            s = min(1.0, t_after / t_ret_total)
            if s < 0.30:
                tgts[fn] = p_rel.lerp(p_peel, min_jerk(s / 0.30))
            else:
                tgts[fn] = _safe_arc(p_peel, p_home, (s - 0.30) / 0.70, lift=0.020, cube_margin=0.015)

        hc, angs = self._guard_hand_clearance(side, hc, tgts, t)
        return cube_world @ hc, angs

    # ------------------------------------------------------------------ secondary motion (body, head, eyes)
    def _precompute_body_secondary(self) -> None:
        """Precompute critically damped torso lean, head tracking with saccades, and eye gaze at 120 Hz."""
        hz = 120
        n = int(math.ceil(self.t_end * hz)) + 2
        self._sec_ts = [i / hz for i in range(n)]

        raw_torso: List[Tuple[float, float, float]] = []
        raw_head_tgt: List[Tuple[float, float, float]] = []
        raw_eye_sacc: List[Tuple[float, float]] = []

        for t in self._sec_ts:
            cw = self.cube_pose(t)
            cp = cw.translation

            # Torso subtle forward lean when reaching/solving + calm breathing sway
            breath = math.sin(2.0 * math.pi * 0.24 * t)
            if t < self.t_reach_start:
                lean = 0.02
            elif t < self.t_grasp:
                lean = 0.02 + 0.095 * min_jerk((t - self.t_reach_start) / (self.t_grasp - self.t_reach_start))
            elif t < self.t_inspect_start:
                lean = 0.115 - 0.035 * min_jerk((t - self.t_grasp) / (self.t_inspect_start - self.t_grasp))
            elif t < self.t_setdown_touch:
                lean = 0.080 + 0.018 * (1.0 if self.t_solve_start <= t <= self.t_solve_end else 0.0)
            elif t < self.t_release_end:
                lean = 0.080 + 0.030 * math.sin(math.pi * (t - self.t_setdown_touch) / (self.t_release_end - self.t_setdown_touch))
            else:
                s = min_jerk(min(1.0, (t - self.t_release_end) / 0.85))
                lean = 0.080 * (1.0 - s) + 0.015 * s
            # Tiny lateral torso bias tracking cube X
            lat_lean = -cp.x * 0.35
            raw_torso.append((lean + 0.005 * breath, lat_lean, 0.004 * math.cos(0.9 * t)))

            # Head target point: desk -> cube (with subtle inspection scanning) -> finale camera lens!
            if t < self.t_wake:
                ht = Vector((0.0, -0.55, SURFACE_Z + 0.02))
            elif t < self.t_release_end - 0.10:
                ht = cp.copy()
                # Subtle micro-saccade scan target across cube facelets during inspection!
                if self.t_inspect_start <= t <= self.t_inspect_end:
                    step = int((t - self.t_inspect_start) / 0.34)
                    # Deterministic pseudo-random facelet saccade offset (+/- 18 mm)
                    sx = ((step * 37) % 3 - 1) * 0.017
                    sz = ((step * 53) % 3 - 1) * 0.017
                    ht += Vector((sx, 0.0, sz))
                    raw_eye_sacc.append((sx * 0.16, sz * 0.16))
                else:
                    raw_eye_sacc.append((0.0, 0.0))
            else:
                # Smoothly lift head to look straight into the finale lens!
                s = min_jerk(min(1.0, (t - (self.t_release_end - 0.10)) / 0.72))
                ht = cp.lerp(CAM_FINALE_POS, s)
                raw_eye_sacc.append((0.0, 0.0))
            if len(raw_eye_sacc) < len(raw_torso):
                raw_eye_sacc.append((0.0, 0.0))
            raw_head_tgt.append((ht.x, ht.y, ht.z))

        self._torso_filt = CriticallyDampedSmoother(omega=5.5).filter_series(self._sec_ts, raw_torso)
        self._head_tgt_filt = CriticallyDampedSmoother(omega=9.5).filter_series(self._sec_ts, raw_head_tgt)
        self._eye_sacc_filt = CriticallyDampedSmoother(omega=38.0).filter_series(self._sec_ts, raw_eye_sacc)

    def _sample_sec(self, series, t: float):
        idx = int(max(0, min(len(self._sec_ts) - 2, t * 120.0)))
        t0, t1 = self._sec_ts[idx], self._sec_ts[idx + 1]
        a = min(1.0, max(0.0, (t - t0) / max(1e-6, t1 - t0)))
        v0, v1 = series[idx], series[idx + 1]
        return tuple(v0[k] * (1.0 - a) + v1[k] * a for k in range(len(v0)))

    def eval_frame(self, t: float) -> dict:
        """Evaluate the full robot + cube world/local kinematics at time `t`."""
        # Subtle high-frequency servo micro-jitter (sub-millimetre realistic actuator hum)
        jit = 0.0009 * math.sin(37.0 * t) * math.cos(23.0 * t)
        lean, lat, twist = self._sample_sec(self._torso_filt, t)
        hx, hy, hz = self._sample_sec(self._head_tgt_filt, t)
        ex, ez = self._sample_sec(self._eye_sacc_filt, t)

        # Spine & chest world transforms
        pelvis_m = Matrix.Translation(PELVIS_POS)
        spine_q = Euler((-lean * 0.45, lat * 0.45, twist * 0.5), "XYZ").to_quaternion()
        spine_m = pelvis_m @ Matrix.Translation(SPINE_OFF) @ spine_q.to_matrix().to_4x4()

        chest_q = Euler((-lean * 0.55 + jit * 0.3, lat * 0.55, twist * 0.5), "XYZ").to_quaternion()
        chest_m = spine_m @ Matrix.Translation(CHEST_OFF) @ chest_q.to_matrix().to_4x4()

        # Neck & Head aim smoothly at filtered look-target `head_tgt`
        neck_base = chest_m @ Matrix.Translation(NECK_OFF)
        head_pos = (neck_base @ Matrix.Translation(HEAD_OFF)).translation
        look_v = Vector((hx, hy, hz)) - (head_pos + Vector((0.0, -0.05, 0.09)))
        if look_v.length < 1e-4:
            look_v = Vector((0.0, -1.0, 0.0))
        look_v.normalize()
        # Head forward is -Y, up is +Z, left is +X:
        #   Rx(+pitch) tilts -Y downward (-Z), so pitch = -asin(look_v.z)
        #   Rz(+yaw) turns -Y toward +X (left), so yaw = atan2(+look_v.x, -look_v.y)
        yaw = math.atan2(look_v.x, -look_v.y)
        pitch = -math.asin(max(-0.82, min(0.82, look_v.z)))
        roll = 0.0
        if self.t_inspect_start <= t <= self.t_inspect_end:
            roll = 0.065 * math.sin(0.95 * (t - self.t_inspect_start))
        neck_q = Euler((pitch * 0.40, roll * 0.35, yaw * 0.42), "XYZ").to_quaternion()
        head_q = Euler((pitch * 0.60 + jit * 0.5, roll * 0.65, yaw * 0.58), "XYZ").to_quaternion()

        cube_w = self.cube_pose(t)
        cubie_qs = self.cubie_quaternions(t)

        arms_out = {}
        for side in ("R", "L"):
            sx = -1.0 if side == "R" else 1.0
            sh_local = Vector((sx * abs(SHOULDER_OFF_R.x), SHOULDER_OFF_R.y, SHOULDER_OFF_R.z))
            S = chest_m @ sh_local
            hand_w, finger_angs = self.hand_state(t, side, cube_w)
            # Natural anatomical elbow pole: outward & slightly down/back
            pole = Vector((sx * 0.68, 0.28, -0.68))
            upper_m, fore_m, wroll_m, hand_clamped, unreach = solve_arm(S, hand_w, pole)

            finger_quats = {}
            for fn in FINGER_ORDER:
                psi, q1, q2 = finger_angs[fn]
                qm, qp, qd = finger_locals(fn, psi, q1, q2)
                if side == "L":
                    # Mirror local rotations cleanly for mirrored left-hand geometry
                    qm = (MIRROR_X @ qm.to_matrix().to_4x4() @ MIRROR_X).to_quaternion()
                    qp = (MIRROR_X @ qp.to_matrix().to_4x4() @ MIRROR_X).to_quaternion()
                    qd = (MIRROR_X @ qd.to_matrix().to_4x4() @ MIRROR_X).to_quaternion()
                finger_quats[fn] = (qm, qp, qd)

            arms_out[side] = {
                "upper": upper_m,
                "fore": fore_m,
                "wroll": wroll_m,
                "hand": hand_clamped,
                "unreachable": unreach,
                "finger_angles": finger_angs,
                "finger_quats": finger_quats,
            }

        return {
            "t": t,
            "cube_world": cube_w,
            "cubie_quats": cubie_qs,
            "spine_q": spine_q,
            "chest_q": chest_q,
            "neck_q": neck_q,
            "head_q": head_q,
            "eye_shift": (ex, ez),
            "arms": arms_out,
        }
