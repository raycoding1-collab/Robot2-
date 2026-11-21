"""Seven-camera cinema package (35-95 mm) with real apertures, focus pulls & handheld drift.

Every cut lands cleanly on physical motion:
  Shot 1 (35 mm, f/2.8): Establishing sunlit lab desk -> KAIRO-04 wakes, reaches & grasps cube.
  Shot 2 (62 mm, f/2.4): Cut on lift -> front-left medium close-up inspecting F, U, B faces.
  Shot 3 (85 mm, f/2.2): Intimate 3/4 portrait catching visor reflection of cube + eye lights + R/L inspect.
  Shot 4 (92 mm, f/2.8): Hero macro-tele on nimble hands & stickerless cube as speedsolve ignites!
  Shot 5 (52 mm, f/2.4): Dynamic low front-3/4 mid-solve showing flying hands, forearms & head focus.
  Shot 6 (95 mm, f/2.8): Crisp close-up of final moves snapping solved + graceful solved presentation.
  Shot 7 (48 mm, f/2.2): Hero finale set-down on ESD mat + cinema rack-focus from cube -> visor eyes!
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import bpy
from mathutils import Vector

from . import geo
from .choreography import Simulation
from .config import CAM_FINALE_POS, CUBE_HOLD_POS, CUBE_REST_POS, SURFACE_Z
from .physics import min_jerk


@dataclass(frozen=True)
class ShotSpec:
    name: str
    t_start: float
    t_end: float
    lens_mm: float
    fstop: float
    pos_start: Vector
    pos_end: Vector
    target_mode: str   # "cube", "head_cube", "hands_macro", "finale_rack"
    drift_amp: float   # subtle handheld/fluid-head micro-drift (metres)


def build_shot_list(sim: Simulation) -> List[ShotSpec]:
    """Define 7 motivated cinema shots tightly locked to physical action beats."""
    t_lift = sim.t_lift_start + 0.12
    t_mid_insp = sim.t_inspect_start + 4.35
    t_pre_solve = sim.t_solve_start - 0.90
    t_mid_solve = sim.t_solve_start + 0.50 * (sim.t_solve_end - sim.t_solve_start)
    t_end_solve = sim.t_solve_end - 0.65
    t_setdown = sim.t_present_end - 0.15

    return [
        ShotSpec(
            name="Cam_01_Establishing",
            t_start=0.0,
            t_end=t_lift,
            lens_mm=34.0,
            fstop=2.8,
            pos_start=Vector((0.68, -1.66, 1.24)),
            pos_end=Vector((0.60, -1.56, 1.21)),
            target_mode="establish",
            drift_amp=0.0022,
        ),
        ShotSpec(
            name="Cam_02_InspectLift",
            t_start=t_lift,
            t_end=t_mid_insp,
            lens_mm=62.0,
            fstop=2.4,
            pos_start=Vector((-0.46, -1.04, 1.09)),
            pos_end=Vector((-0.42, -0.98, 1.11)),
            target_mode="cube_upper",
            drift_amp=0.0018,
        ),
        ShotSpec(
            name="Cam_03_VisorInspect",
            t_start=t_mid_insp,
            t_end=t_pre_solve,
            lens_mm=50.0,
            fstop=2.2,
            # Classic cinema 3/4 two-shot: sweeping glossy visor (upper-left) + hands inspecting cube (lower-right)!
            pos_start=Vector((0.68, -1.24, 1.15)),
            pos_end=Vector((0.60, -1.16, 1.15)),
            target_mode="head_cube",
            drift_amp=0.0016,
        ),
        ShotSpec(
            name="Cam_04_SolveHeroMacro",
            t_start=t_pre_solve,
            t_end=t_mid_solve,
            lens_mm=92.0,
            fstop=2.8,
            pos_start=Vector((0.20, -0.82, 0.99)),
            pos_end=Vector((0.16, -0.79, 0.98)),
            target_mode="hands_macro",
            drift_amp=0.0014,
        ),
        ShotSpec(
            name="Cam_05_MidSolveDynamic",
            t_start=t_mid_solve,
            t_end=t_end_solve,
            lens_mm=54.0,
            fstop=2.4,
            pos_start=Vector((-0.44, -1.04, 0.96)),
            pos_end=Vector((-0.38, -0.99, 0.99)),
            target_mode="cube_torso",
            drift_amp=0.0020,
        ),
        ShotSpec(
            name="Cam_06_FinishPresent",
            t_start=t_end_solve,
            t_end=t_setdown,
            lens_mm=95.0,
            fstop=2.8,
            pos_start=Vector((-0.18, -0.86, 1.00)),
            pos_end=Vector((0.06, -0.88, 1.02)),
            target_mode="hands_macro",
            drift_amp=0.0015,
        ),
        ShotSpec(
            name="Cam_07_FinaleLensLook",
            t_start=t_setdown,
            t_end=sim.t_end + 0.1,
            lens_mm=38.0,
            fstop=2.2,
            pos_start=CAM_FINALE_POS + Vector((0.08, -0.14, 0.02)),
            pos_end=CAM_FINALE_POS + Vector((0.04, -0.10, 0.00)),
            target_mode="finale_rack",
            drift_amp=0.0018,
        ),
    ]


def _handheld_drift(t: float, seed: float, amp: float) -> Vector:
    """Organic multi-harmonic camera operator breathing/micro-drift (never robotic linear)."""
    dx = amp * (0.60 * math.sin(0.83 * t + seed) + 0.40 * math.sin(1.97 * t + 2.1 * seed))
    dy = amp * 0.5 * (0.55 * math.cos(0.71 * t + 1.3 * seed) + 0.45 * math.sin(1.63 * t + seed))
    dz = amp * (0.65 * math.cos(0.67 * t + 0.7 * seed) + 0.35 * math.sin(2.29 * t + 1.7 * seed))
    return Vector((dx, dy, dz))


def evaluate_camera_at(shot: ShotSpec, sim: Simulation, t: float, seed: float) -> Tuple[Vector, Vector, float]:
    """Return (cam_pos, aim_point, focus_distance) at time `t`."""
    u = min_jerk(min(1.0, max(0.0, (t - shot.t_start) / max(1e-4, shot.t_end - shot.t_start))))
    pos = shot.pos_start.lerp(shot.pos_end, u) + _handheld_drift(t, seed, shot.drift_amp)

    cube_p = sim.cube_pose(t).translation
    head_p = Vector((0.0, -0.05, 1.30))

    if shot.target_mode == "establish":
        aim = cube_p.lerp(head_p, 0.52) + Vector((0.0, 0.0, 0.02))
        focus_p = cube_p.lerp(head_p, 0.42)
    elif shot.target_mode == "cube_upper":
        aim = cube_p + Vector((0.0, 0.02, 0.04))
        focus_p = cube_p
    elif shot.target_mode == "head_cube":
        # Frame BOTH wide glossy visor (upper-left) and hands inspecting the stickerless cube (lower-right)!
        aim = head_p.lerp(cube_p, 0.58) + Vector((-0.02, -0.04, 0.01))
        focus_p = head_p.lerp(cube_p, 0.25 + 0.45 * u)
    elif shot.target_mode == "hands_macro":
        aim = cube_p + Vector((0.0, 0.0, -0.004))
        focus_p = cube_p
    elif shot.target_mode == "cube_torso":
        aim = cube_p.lerp(head_p, 0.34)
        focus_p = cube_p
    elif shot.target_mode == "finale_rack":
        aim = cube_p.lerp(head_p, 0.54) + Vector((0.0, 0.0, 0.04))
        if t < sim.t_setdown_touch + 0.35:
            rack = 0.0
        else:
            rack = min_jerk(min(1.0, (t - (sim.t_setdown_touch + 0.35)) / 0.85))
        focus_p = cube_p.lerp(head_p + Vector((0.0, -0.06, 0.01)), rack)
    else:
        aim = cube_p
        focus_p = cube_p

    # Tiny operator pan micro-drift
    aim = aim + _handheld_drift(t, seed + 5.7, shot.drift_amp * 0.45)
    focus_dist = max(0.20, (focus_p - pos).length)
    return pos, aim, focus_dist


def build_and_animate_cameras(coll, sim: Simulation) -> Tuple[List[ShotSpec], Dict[str, bpy.types.Object]]:
    """Create all 7 physical cinema cameras, keyframe them, and bind timeline cut markers."""
    sc = bpy.context.scene
    sc.timeline_markers.clear()
    shots = build_shot_list(sim)
    cam_objs: Dict[str, bpy.types.Object] = {}

    for idx, shot in enumerate(shots):
        cd = bpy.data.cameras.new(shot.name)
        cd.sensor_width = 36.0
        cd.lens = shot.lens_mm
        cd.clip_start = 0.05
        cd.clip_end = 50.0
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = shot.fstop
        cd.dof.aperture_blades = 7
        cd.dof.aperture_rotation = math.radians(15.0)
        co = bpy.data.objects.new(shot.name, cd)
        co.rotation_mode = "QUATERNION"
        geo.link_obj(co, coll)
        cam_objs[shot.name] = co

        f_start = max(1, int(round(shot.t_start * sim.fps)) + 1)
        f_end = min(sim.total_frames, int(round(shot.t_end * sim.fps)) + 1)
        # Bind marker on the exact cut frame
        mk = sc.timeline_markers.new(f"Cut_{idx + 1:02d}_{shot.name}", frame=f_start)
        mk.camera = co

        # Keyframe every frame in the shot (+1 pad frame on each side for clean motion blur)
        seed = 1.73 * (idx + 1)
        for f in range(max(1, f_start - 1), min(sim.total_frames, f_end + 1) + 1):
            t = (f - 1) / sim.fps
            pos, aim, fdist = evaluate_camera_at(shot, sim, t, seed)
            co.location = pos
            co.rotation_quaternion = (aim - pos).to_track_quat("-Z", "Y")
            cd.dof.focus_distance = fdist
            co.keyframe_insert("location", frame=f)
            co.keyframe_insert("rotation_quaternion", frame=f)
            cd.dof.keyframe_insert("focus_distance", frame=f)

    sc.camera = cam_objs[shots[0].name]
    return shots, cam_objs


def active_camera_at(shots: List[ShotSpec], sim: Simulation, t: float) -> Tuple[int, ShotSpec, Vector]:
    """Return (shot_index, ShotSpec, camera_world_pos) active at time `t` (for sound mic tracking)."""
    for i, s in enumerate(shots):
        if s.t_start <= t < s.t_end:
            pos, _, _ = evaluate_camera_at(s, sim, t, 1.73 * (i + 1))
            return i, s, pos
    s = shots[-1]
    pos, _, _ = evaluate_camera_at(s, sim, t, 1.73 * len(shots))
    return len(shots) - 1, s, pos
