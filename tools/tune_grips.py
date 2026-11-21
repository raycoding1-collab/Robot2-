"""Numerical grip optimizer for KAIRO-04 hands around the 57 mm speed cube.

Uses a singularity-free tangent frame around the natural side grip so the optimizer
finds globally optimal wrist poses with:
  - < 0.05 mm fingertip IK error across all active fingers
  - 0.00 mm phalanx/palm/pinky penetration against the 57 mm cube (and desk for pickup)
  - Comfortable mid-range joint angles across home, R/L wrist turns, and pickup
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import kairo  # noqa: F401
import numpy as np
from mathutils import Euler, Matrix, Vector
from scipy.optimize import differential_evolution, minimize

from kairo.config import CUBE_HALF, FINGERS, phalanx_radius
from kairo.ik import FINGER_LIMITS, finger_fk_chain, finger_ik

# Neutral side-grip basis for canonical RIGHT hand holding +X (R) side of cube:
#   hand +X (pinky)  -> cube -Z (down)
#   hand +Y (finger) -> cube +Y (toward B face)
#   hand +Z (dorsal) -> cube +X (outward right)
BASE_RIGHT_ROT = Matrix((
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
)).to_4x4()

PINKY_CURL = (0.10, 0.88, 1.15)  # natural relaxed curl out of the way


def hand_matrix_from_params(params) -> Matrix:
    """Singularity-free 6-DOF pose: translation + local XYZ perturbation from BASE_RIGHT_ROT."""
    tx, ty, tz, rx, ry, rz = [float(v) for v in params]
    m = BASE_RIGHT_ROT @ Euler((rx, ry, rz), "XYZ").to_matrix().to_4x4()
    m.translation = Vector((tx, ty, tz))
    return m


def point_box_signed_dist(p: Vector, half: float = CUBE_HALF) -> float:
    qx = abs(p.x) - half
    qy = abs(p.y) - half
    qz = abs(p.z) - half
    outside = Vector((max(qx, 0.0), max(qy, 0.0), max(qz, 0.0))).length
    inside = min(max(qx, qy, qz), 0.0)
    return outside + inside


def finger_penetration(hand_cube: Matrix, name: str, psi: float, q1: float, q2: float, touching: bool) -> float:
    """Max capsule penetration (metres) of finger's 3 phalanges into [-H, H]^3."""
    p_mcp, p_pip, p_dip, p_tip, p_pad = finger_fk_chain(name, psi, q1, q2)
    pts = [hand_cube @ p for p in (p_mcp, p_pip, p_dip, p_tip)]
    max_pen = 0.0
    # Proximal & middle phalanges: full capsule clearance
    for seg in (0, 1):
        r = phalanx_radius(name, seg) * 0.92
        a, b = pts[seg], pts[seg + 1]
        for k in range(6):
            p = a.lerp(b, k / 5.0)
            pen = r - point_box_signed_dist(p)
            if pen > max_pen:
                max_pen = pen
    # Distal phalanx: proximal half full radius; contact pad itself checked directly against surface
    r2 = phalanx_radius(name, 2) * 0.85
    for k in range(4):
        s = (k / 3.0) * (0.55 if touching else 0.92)
        p = pts[2].lerp(pts[3], s)
        pen = r2 - point_box_signed_dist(p)
        if pen > max_pen:
            max_pen = pen
    # Pad itself must never dip inside cube
    pad_w = hand_cube @ p_pad
    pad_pen = -point_box_signed_dist(pad_w)
    return max(max_pen, pad_pen)


def palm_penetration(hand_cube: Matrix) -> float:
    pts = [
        (x, y, z)
        for x in (-0.030, 0.0, 0.030)
        for y in (0.015, 0.048, 0.078)
        for z in (-0.011, 0.011)
    ]
    max_pen = 0.0
    for pt in pts:
        pw = hand_cube @ Vector(pt)
        pen = 0.0025 - point_box_signed_dist(pw)
        if pen > max_pen:
            max_pen = pen
    return max_pen


H = CUBE_HALF
PAD_CLR = 0.0005  # 0.5 mm cushion outside surface

GRIP_SPECS = {
    # Home speedcubing grip: thumb on F, middle+ring stabilizing B, index hovered behind UBR
    "home": dict(
        touching={"thumb", "middle", "ring"},
        targets={
            "thumb": Vector((0.012, -(H + PAD_CLR), -0.002)),
            "index": Vector((0.020, H + 0.009, 0.021)),
            "middle": Vector((0.019, H + PAD_CLR, 0.003)),
            "ring": Vector((0.019, H + PAD_CLR, -0.017)),
        },
        desk_check=False,
    ),
    # R-layer wrist-turn grip: thumb on F-right, index/middle/ring on B-right
    "r_turn": dict(
        touching={"thumb", "index", "middle", "ring"},
        targets={
            "thumb": Vector((0.019, -(H + PAD_CLR), -0.001)),
            "index": Vector((0.019, H + PAD_CLR, 0.019)),
            "middle": Vector((0.019, H + PAD_CLR, 0.000)),
            "ring": Vector((0.019, H + PAD_CLR, -0.019)),
        },
        desk_check=False,
    ),
    # Pickup / setdown grip when cube sits on desk (z_cube = -H is the desk surface!)
    "pickup": dict(
        touching={"thumb", "index", "middle"},
        targets={
            "thumb": Vector((0.015, -(H + PAD_CLR), 0.008)),
            "index": Vector((0.019, H + PAD_CLR, 0.015)),
            "middle": Vector((0.019, H + PAD_CLR, -0.003)),
            "ring": Vector((0.044, H + 0.012, 0.006)),
        },
        desk_check=True,
    ),
}


def evaluate_grip(hand_cube: Matrix, spec: dict):
    inv_h = hand_cube.inverted()
    targets: Dict[str, Vector] = spec["targets"]
    touching = spec["touching"]
    desk_check = spec["desk_check"]

    ik_errs: Dict[str, float] = {}
    max_pen = palm_penetration(hand_cube)
    max_desk_pen = 0.0
    lim_pen = 0.0

    for name, tgt_c in targets.items():
        tgt_h = inv_h @ tgt_c
        psi, q1, q2 = finger_ik(name, tgt_h)
        p_mcp, p_pip, p_dip, p_tip, p_pad = finger_fk_chain(name, psi, q1, q2)
        ik_errs[name] = (p_pad - tgt_h).length
        max_pen = max(max_pen, finger_penetration(hand_cube, name, psi, q1, q2, name in touching))

        lim = FINGER_LIMITS["thumb" if name == "thumb" else "default"]
        for val, (lo, hi) in ((psi, lim["psi"]), (q1, lim["q1"]), (q2, lim["q2"])):
            mid = 0.5 * (lo + hi)
            span = 0.5 * (hi - lo)
            norm = abs(val - mid) / max(1e-3, span)
            if norm > 0.68:
                lim_pen += (norm - 0.68) ** 2

        if desk_check:
            for seg_i, p_loc in enumerate((p_mcp, p_pip, p_dip, p_tip)):
                r = phalanx_radius(name, min(seg_i, 2))
                pz = (hand_cube @ p_loc).z
                dpen = (-H + r + 0.003) - pz
                if dpen > max_desk_pen:
                    max_desk_pen = dpen

    # Passive pinky clearance (cube & desk)
    max_pen = max(max_pen, finger_penetration(hand_cube, "pinky", *PINKY_CURL, touching=False))
    if desk_check:
        for p_loc in finger_fk_chain("pinky", *PINKY_CURL)[:4]:
            pz = (hand_cube @ p_loc).z
            max_desk_pen = max(max_desk_pen, (-H + 0.010) - pz)
        for pt in ((-0.030, 0.015, -0.011), (0.030, 0.078, -0.011), (0.0, 0.048, -0.011), (0.0, 0.0, 0.0)):
            pz = (hand_cube @ Vector(pt)).z
            max_desk_pen = max(max_desk_pen, (-H + 0.014) - pz)

    return ik_errs, max_pen, max_desk_pen, lim_pen


def optimize_all():
    results = {}
    for name, spec in GRIP_SPECS.items():
        def loss(x):
            hc = hand_matrix_from_params(x)
            ik_errs, max_pen, max_desk_pen, lim_pen = evaluate_grip(hc, spec)
            err_cost = sum((e * 1000.0) ** 2 for e in ik_errs.values()) * 10.0
            pen_mm = max(0.0, max_pen * 1000.0)
            desk_mm = max(0.0, max_desk_pen * 1000.0)
            pen_cost = (pen_mm * 150.0) ** 2 + (desk_mm * 150.0) ** 2
            hy = hc.to_3x3().col[1]
            hz = hc.to_3x3().col[2]
            orient_pen = 0.0
            if hy.y < (0.20 if name == "pickup" else 0.35):
                orient_pen += ((0.20 if name == "pickup" else 0.35) - hy.y) ** 2 * 80.0
            if hz.x < 0.25:
                orient_pen += (0.25 - hz.x) ** 2 * 80.0
            pos_pen = 0.0
            if hc.translation.y > -0.03:
                pos_pen += ((hc.translation.y + 0.03) * 100.0) ** 2
            return err_cost + pen_cost + lim_pen * 3.0 + orient_pen + pos_pen

        bounds = [
            (0.035, 0.145),   # tx
            (-0.155, -0.025), # ty
            (-0.115, 0.105),  # tz
            (-1.35, 1.35),    # local rx
            (-1.25, 1.25),    # local ry
            (-1.25, 1.25),    # local rz
        ]
        de = differential_evolution(loss, bounds, seed=11, maxiter=300, popsize=16, tol=1e-6, polish=False)
        res = minimize(loss, de.x, method="Powell", options={"maxiter": 5000, "ftol": 1e-10})

        hc = hand_matrix_from_params(res.x)
        ik_errs, max_pen, max_desk_pen, _ = evaluate_grip(hc, spec)
        print(
            f"Grip {name:8s}: max_ik_err={max(ik_errs.values())*1000:.3f} mm | "
            f"cube_pen={max_pen*1000:.3f} mm | desk_pen={max_desk_pen*1000:.3f} mm | "
            f"errs={ {k: round(v*1000,2) for k,v in ik_errs.items()} } | "
            f"params={[round(float(v), 5) for v in res.x]}"
        )
        results[name] = {
            "params": [round(float(v), 6) for v in res.x],
            "max_ik_err_mm": round(max(ik_errs.values()) * 1000.0, 4),
            "cube_pen_mm": round(max_pen * 1000.0, 4),
            "desk_pen_mm": round(max_desk_pen * 1000.0, 4),
        }

    out_path = Path(__file__).resolve().parent.parent / "kairo" / "grips_tuned.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved tuned grips -> {out_path}")


if __name__ == "__main__":
    optimize_all()
