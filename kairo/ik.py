"""Analytic IK for KAIRO-04 arms (2-bone + forearm twist) and coupled 3-link fingers.

Also provides exact forward kinematics for every phalanx segment so collision and
penetration checks operate on real 3D capsule geometry, not just fingertip targets.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

from mathutils import Matrix, Quaternion, Vector

from .config import FINGERS, FORE_LEN, UPPER_LEN, WRIST_ROLL_AT, finger_rest_frame, pad_point


def make_frame(x: Vector, y: Vector, origin: Vector) -> Matrix:
    """Build 4x4 orthonormal frame with +X, +Y axes (Z = X x Y) at `origin`."""
    x = x.normalized()
    y = (y - x * y.dot(x)).normalized()
    z = x.cross(y)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = origin
    return m


def solve_arm(S: Vector, hand_world: Matrix, pole: Vector) -> Tuple[Matrix, Matrix, Matrix, Matrix, bool]:
    """Analytic 2-bone arm IK with pole vector and forearm pronation/supination roll.

    Local bone frames: +Y along limb distally, +X = elbow flexion hinge axis.
    Returns (upper_world, forearm_world, wrist_roll_world, hand_world_clamped, unreachable).
    """
    W = hand_world.translation.copy()
    L1, L2 = UPPER_LEN, FORE_LEN
    d = W - S
    dist = d.length
    reach = (L1 + L2) * 0.9992
    unreachable = dist > reach
    dist_c = min(max(dist, abs(L1 - L2) + 1e-4), reach)
    k = d.normalized()

    cos_a = (L1 * L1 + dist_c * dist_c - L2 * L2) / (2.0 * L1 * dist_c)
    a = math.acos(max(-1.0, min(1.0, cos_a)))

    p = pole - k * pole.dot(k)
    if p.length < 1e-6:
        p = Vector((0.0, 0.0, -1.0)) - k * k.z
    p.normalize()

    E = S + L1 * (math.cos(a) * k + math.sin(a) * p)
    Wc = S + k * dist_c

    up_dir = (E - S).normalized()
    fo_dir = (Wc - E).normalized()
    hinge = up_dir.cross(fo_dir)
    if hinge.length < 1e-6:
        hinge = up_dir.cross(p)
    hinge.normalize()

    upper = make_frame(hinge, up_dir, S)
    fore = make_frame(hinge, fo_dir, E)

    # Distribute forearm pronation/supination roll smoothly toward hand orientation
    rel = fore.to_3x3().transposed() @ hand_world.to_3x3()
    xh = rel.col[0]
    roll = math.atan2(-xh.z, xh.x)
    wroll = fore @ Matrix.Translation((0.0, WRIST_ROLL_AT, 0.0)) @ Matrix.Rotation(roll, 4, "Y")

    hand = hand_world.copy()
    if unreachable:
        hand.translation = Wc
    return upper, fore, wroll, hand, unreachable


# ------------------------------------------------------------------ coupled 3-link finger IK
_REST_CACHE: Dict[str, Matrix] = {}

FINGER_LIMITS = {
    "thumb": dict(psi=(-0.72, 0.92), q1=(-0.48, 1.22), q2=(-0.08, 1.42)),
    "default": dict(psi=(-0.28, 0.28), q1=(-0.32, 1.62), q2=(0.0, 1.72)),
}


def rest_frame(name: str) -> Matrix:
    if name not in _REST_CACHE:
        _REST_CACHE[name] = finger_rest_frame(name)
    return _REST_CACHE[name]


def _planar_pad(name: str, q1: float, q2: float) -> Tuple[float, float]:
    """2D (u=forward, v=palmar) pad coordinates in the finger's flexion plane."""
    f = FINGERS[name]
    l1, l2, _ = f["lens"]
    q3 = f["couple"] * q2
    pp = pad_point(name)
    along, depth = pp.y, -pp.z
    u = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
    v = l1 * math.sin(q1) + l2 * math.sin(q1 + q2)
    phi = q1 + q2 + q3
    u += along * math.cos(phi) - depth * math.sin(phi)
    v += along * math.sin(phi) + depth * math.cos(phi)
    return u, v


def finger_ik(name: str, target_hand: Vector) -> Tuple[float, float, float]:
    """Solve (psi, q1, q2) in canonical right-hand space so fingertip pad hits `target_hand`."""
    f = FINGERS[name]
    lim = FINGER_LIMITS["thumb" if name == "thumb" else "default"]
    R = rest_frame(name)
    T = R.transposed() @ (target_hand - Vector(f["base"]))

    # Choose abduction angle psi aligning flexion plane with target
    cands = (math.atan2(-T.x, T.y), math.atan2(T.x, -T.y))
    best_cost = 1e9
    psi = 0.0
    for c in cands:
        cc = max(lim["psi"][0], min(lim["psi"][1], c))
        cost = abs(cc - c) * 12.0 + abs(cc)
        if cost < best_cost:
            best_cost = cost
            psi = cc

    u = -T.x * math.sin(psi) + T.y * math.cos(psi)
    v = -T.z
    rho = math.hypot(u, v)

    lo, hi = lim["q2"]

    def reach_dist(q2_val: float) -> float:
        a, b = _planar_pad(name, 0.0, q2_val)
        return math.hypot(a, b)

    if rho >= reach_dist(lo):
        q2 = lo
    elif rho <= reach_dist(hi):
        q2 = hi
    else:
        a_, b_ = lo, hi
        for _ in range(42):
            mid = 0.5 * (a_ + b_)
            if reach_dist(mid) > rho:
                a_ = mid
            else:
                b_ = mid
        q2 = 0.5 * (a_ + b_)

    pu, pv = _planar_pad(name, 0.0, q2)
    q1 = math.atan2(v, u) - math.atan2(pv, pu)
    q1 = (q1 + math.pi) % (2.0 * math.pi) - math.pi
    q1 = max(lim["q1"][0], min(lim["q1"][1], q1))
    return psi, q1, q2


def finger_locals(name: str, psi: float, q1: float, q2: float) -> Tuple[Quaternion, Quaternion, Quaternion]:
    """Local rotation quaternions for MCP, PIP, DIP joints in canonical right-hand space."""
    f = FINGERS[name]
    R = rest_frame(name).to_4x4()
    m1 = R @ Matrix.Rotation(psi, 4, "Z") @ Matrix.Rotation(-q1, 4, "X")
    m2 = Matrix.Rotation(-q2, 4, "X")
    m3 = Matrix.Rotation(-f["couple"] * q2, 4, "X")
    return m1.to_quaternion(), m2.to_quaternion(), m3.to_quaternion()


def finger_fk_chain(name: str, psi: float, q1: float, q2: float) -> Tuple[Vector, Vector, Vector, Vector, Vector]:
    """Forward kinematics in canonical hand space -> (p_mcp, p_pip, p_dip, p_tip, p_pad)."""
    f = FINGERS[name]
    l1, l2, l3 = f["lens"]
    q_1, q_2, q_3 = finger_locals(name, psi, q1, q2)

    m1 = Matrix.Translation(f["base"]) @ q_1.to_matrix().to_4x4()
    p_mcp = m1.translation.copy()

    m2 = m1 @ Matrix.Translation((0.0, l1, 0.0)) @ q_2.to_matrix().to_4x4()
    p_pip = m2.translation.copy()

    m3 = m2 @ Matrix.Translation((0.0, l2, 0.0)) @ q_3.to_matrix().to_4x4()
    p_dip = m3.translation.copy()

    p_tip = m3 @ Vector((0.0, l3, 0.0))
    p_pad = m3 @ pad_point(name)
    return p_mcp, p_pip, p_dip, p_tip, p_pad
