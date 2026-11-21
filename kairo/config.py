"""Shared physical dimensions, rig constants, and film timing (SI units: metres, seconds).

Coordinate conventions:
  World: Z up, robot seated at origin facing -Y. Robot's RIGHT = -X, LEFT = +X.
  Cube local: +X=R (Red), -X=L (Orange), -Y=F (Green), +Y=B (Blue), +Z=U (White), -Z=D (Yellow).
  Upright cube facing the robot therefore has world rotation Rz(180 deg).
"""
from __future__ import annotations

import math
from mathutils import Matrix, Vector

FPS = 24

# ------------------------------------------------------------------ 3x3 speed cube (57.0 mm)
CUBIE_PITCH = 0.0190          # 19.0 mm pitch -> 3 * 19 = 57.0 mm exact cube
CUBE_SIZE = 3.0 * CUBIE_PITCH # 57.0 mm
CUBE_HALF = 0.5 * CUBE_SIZE   # 28.5 mm
CUBIE_SIZE = 0.01886          # 0.14 mm precision clearance gap between layers
CUBIE_BEVEL = 0.00185         # crisp rounded speedcube bevel
CAP_INSET = 0.00035           # subtle stickerless cap parting groove depth
FLORIAN_RADIUS = 0.0038       # inner corner-cutting Florian relief radius

# ------------------------------------------------------------------ workbench & rest pose
TABLE_TOP_Z = 0.740
MAT_THICK = 0.0025            # 2.5 mm ESD desk mat
SURFACE_Z = TABLE_TOP_Z + MAT_THICK
CUBE_REST_POS = Vector((0.0, -0.355, SURFACE_Z + CUBE_HALF))
CUBE_HOLD_POS = Vector((0.0, -0.315, SURFACE_Z + 0.165))
CUBE_UPRIGHT = Matrix.Rotation(math.pi, 4, "Z")

# ------------------------------------------------------------------ KAIRO-04 skeleton (human proportions)
PELVIS_POS = Vector((0.0, 0.035, 0.595))
SPINE_OFF = Vector((0.0, 0.0, 0.105))
CHEST_OFF = Vector((0.0, 0.0, 0.155))
NECK_OFF = Vector((0.0, 0.004, 0.332))
HEAD_OFF = Vector((0.0, -0.006, 0.088))
SHOULDER_OFF_R = Vector((-0.204, 0.010, 0.278))  # right shoulder in chest frame; left mirrors X

UPPER_LEN = 0.286
FORE_LEN = 0.264
WRIST_ROLL_AT = 0.165

EYE_OFFSET = Vector((0.0, -0.098, 0.102))        # head-local optic midpoint

# Canonical RIGHT hand frame:
#   +Y = distal (toward fingers), +Z = dorsal (back of hand), +X = ulnar (pinky side), -X = radial (thumb side)
# Proportions tuned for slender, sculpted dexterous fingers around a 57 mm speed cube.
FINGERS = {
    "index": dict(
        base=(-0.0255, 0.085, 0.001),
        splay=math.radians(-4.5),
        lens=(0.043, 0.0255, 0.0225),
        width=0.0148,
        thick=0.0132,
        couple=0.70,
    ),
    "middle": dict(
        base=(-0.0070, 0.090, 0.002),
        splay=math.radians(0.0),
        lens=(0.047, 0.0285, 0.0235),
        width=0.0152,
        thick=0.0136,
        couple=0.70,
    ),
    "ring": dict(
        base=(0.0110, 0.087, 0.001),
        splay=math.radians(4.5),
        lens=(0.044, 0.0265, 0.0225),
        width=0.0144,
        thick=0.0130,
        couple=0.70,
    ),
    "pinky": dict(
        base=(0.0275, 0.079, -0.001),
        splay=math.radians(10.5),
        lens=(0.034, 0.0205, 0.0195),
        width=0.0128,
        thick=0.0118,
        couple=0.70,
    ),
    "thumb": dict(
        base=(-0.0235, 0.021, -0.010),
        dir=(-0.58, 0.73, -0.36),
        nail=(-0.55, -0.04, 0.84),
        lens=(0.041, 0.0305, 0.0260),
        width=0.0172,
        thick=0.0148,
        couple=0.82,
    ),
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

MIRROR_X = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))

# Final camera position so KAIRO-04's eyes meet the lens at the finale
CAM_FINALE_POS = Vector((0.52, -1.48, 1.18))


def finger_rest_frame(name: str) -> Matrix:
    """3x3 rest rotation (columns: lateral X, forward Y, dorsal Z) at finger MCP/CMC base."""
    f = FINGERS[name]
    if name == "thumb":
        fwd = Vector(f["dir"]).normalized()
        dor = Vector(f["nail"])
        dor = (dor - fwd * dor.dot(fwd)).normalized()
    else:
        fwd = (Matrix.Rotation(-f["splay"], 3, "Z") @ Vector((0.0, 1.0, 0.0))).normalized()
        dor = Vector((0.0, 0.0, 1.0))
    lat = fwd.cross(dor).normalized()
    return Matrix((lat, fwd, dor)).transposed()


def pad_point(name: str) -> Vector:
    """Silicone fingertip contact pad center in distal phalanx local frame."""
    f = FINGERS[name]
    L = f["lens"][2]
    t = f["thick"] * 0.84
    return Vector((0.0, 0.68 * L, -0.54 * t))


def phalanx_radius(name: str, seg_idx: int) -> float:
    """Conservative collision radius for each phalanx segment (0=prox, 1=mid, 2=distal)."""
    f = FINGERS[name]
    taper = (1.0, 0.90, 0.80)[seg_idx]
    return 0.5 * f["thick"] * taper
