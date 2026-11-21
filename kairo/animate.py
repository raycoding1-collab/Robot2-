"""Bake the physically simulated cube & KAIRO-04 motion into Blender fcurves + audio telemetry."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import bpy
from mathutils import Matrix, Quaternion, Vector

from .cameras import active_camera_at, build_shot_list
from .choreography import Simulation
from .config import EYE_OFFSET, FINGER_ORDER


def _ensure_continuous_quat(q: Quaternion, prev: Quaternion | None) -> Quaternion:
    """Avoid quaternion antipodal double-cover flips so sub-frame motion blur never spins 360 deg!"""
    if prev is not None and q.dot(prev) < 0.0:
        return -q
    return q


def _set_loc_quat(obj, m: Matrix, frame: float, prev_q: Dict[str, Quaternion]):
    obj.location = m.translation
    q = _ensure_continuous_quat(m.to_quaternion(), prev_q.get(obj.name))
    prev_q[obj.name] = q.copy()
    obj.rotation_quaternion = q
    obj.keyframe_insert("location", frame=frame)
    obj.keyframe_insert("rotation_quaternion", frame=frame)


def _set_quat(obj, q: Quaternion, frame: float, prev_q: Dict[str, Quaternion]):
    qc = _ensure_continuous_quat(q, prev_q.get(obj.name))
    prev_q[obj.name] = qc.copy()
    obj.rotation_quaternion = qc
    obj.keyframe_insert("rotation_quaternion", frame=frame)


def bake_animation(sim: Simulation, rig: dict, cube_root, cubies: List[bpy.types.Object]) -> dict:
    """Bake every frame (+ half-frame subsamples during the rapid solve for razor-accurate motion blur!)."""
    sc = bpy.context.scene
    sc.frame_start = 1
    sc.frame_end = sim.total_frames
    prev_q: Dict[str, Quaternion] = {}

    # Sample frames at integer frames everywhere, PLUS 0.5-frame sub-keys during the rapid solve
    # so Cycles' 180-degree shutter traces true curved layer flicks & magnetic overshoot!
    f_solve0 = int(sim.t_solve_start * sim.fps) - 2
    f_solve1 = int(sim.t_solve_end * sim.fps) + 4
    sample_frames: List[float] = []
    for f in range(1, sim.total_frames + 1):
        sample_frames.append(float(f))
        if f_solve0 <= f <= f_solve1 and f < sim.total_frames:
            sample_frames.append(f + 0.5)

    for fr in sample_frames:
        t = (fr - 1.0) / sim.fps
        st = sim.eval_frame(t)

        # Cube root + 26 cubies
        _set_loc_quat(cube_root, st["cube_world"], fr, prev_q)
        for ci, c_obj in enumerate(cubies):
            _set_quat(c_obj, st["cubie_quats"][ci], fr, prev_q)

        # Torso, neck, head, articulated optic eyes
        _set_quat(rig["spine"], st["spine_q"], fr, prev_q)
        _set_quat(rig["chest"], st["chest_q"], fr, prev_q)
        _set_quat(rig["neck"], st["neck_q"], fr, prev_q)
        _set_quat(rig["head"], st["head_q"], fr, prev_q)

        ex, ez = st["eye_shift"]
        for eye_o, sx in zip(rig["eyes"], (-1.0, 1.0)):
            eye_o.location = (sx * 0.0265 + ex, -0.0915, 0.112 + ez)
            eye_o.keyframe_insert("location", frame=fr)

        # Arms & fingers
        for side in ("R", "L"):
            a_st = st["arms"][side]
            a_rig = rig["arms"][side]
            _set_loc_quat(a_rig["upper"], a_st["upper"], fr, prev_q)
            _set_loc_quat(a_rig["fore"], a_st["fore"], fr, prev_q)
            _set_loc_quat(a_rig["wroll"], a_st["wroll"], fr, prev_q)
            _set_loc_quat(a_rig["hand"], a_st["hand"], fr, prev_q)
            for fn in FINGER_ORDER:
                qm, qp, qd = a_st["finger_quats"][fn]
                mcp, pip, dip = a_rig["fingers"][fn]
                _set_quat(mcp, qm, fr, prev_q)
                _set_quat(pip, qp, fr, prev_q)
                _set_quat(dip, qd, fr, prev_q)

    # Export dense physical telemetry (200 Hz) so make_sound.py reads the EXACT same simulation!
    return export_audio_telemetry(sim)


def export_audio_telemetry(sim: Simulation, out_path: Path | None = None) -> dict:
    """Sample physics velocities, joint speeds, and camera mic perspective at 200 Hz for sound synthesis."""
    hz = 200
    n = int(round(sim.t_end * hz)) + 1
    dt = 1.0 / hz
    shots = build_shot_list(sim)

    layer_speed = [0.0] * n
    servo_arm = [0.0] * n
    servo_wrist = [0.0] * n
    servo_finger = [0.0] * n
    servo_head = [0.0] * n
    cam_dist = [0.0] * n
    cam_pan = [0.0] * n
    cam_shot = [0] * n
    fan_load = [0.0] * n

    prev_st = None
    for i in range(n):
        t = i * dt
        # Instantaneous cube layer angular speed (rad/s)
        w_lay = 0.0
        for ev in sim.turns:
            if ev.t_start <= t <= ev.t_end:
                w_lay += abs(ev.profile.vel(t - ev.t_start))
        layer_speed[i] = round(w_lay, 3)

        # Active camera distance & stereo pan relative to cube
        s_idx, s_spec, cpos = active_camera_at(shots, sim, t)
        cam_shot[i] = s_idx
        cube_p = sim.cube_pose(t).translation
        vec = cube_p - cpos
        dist = max(0.25, vec.length)
        cam_dist[i] = round(dist, 4)
        # Camera right axis roughly (-vec.y, vec.x, 0) normalized -> horizontal stereo pan
        fwd_xy = Vector((vec.x, vec.y, 0.0)).normalized()
        right_xy = Vector((fwd_xy.y, -fwd_xy.x, 0.0))
        # Subtle offset from cube center to active hand when solving
        src_p = cube_p.copy()
        for ev in sim.turns:
            if ev.t_start <= t <= ev.t_end:
                src_p.x += -0.04 if ev.hand == "R" else 0.04
                break
        pan = max(-0.7, min(0.7, (src_p - cpos).dot(right_xy) / max(0.3, dist) * 1.8))
        cam_pan[i] = round(pan, 4)

        # Cooling fan gentle spin-up during analytical inspection & solve
        if t < sim.t_wake:
            fl = 0.0
        elif t < sim.t_inspect_start:
            fl = 0.30 * min(1.0, (t - sim.t_wake) / 1.4)
        elif t < sim.t_solve_end:
            fl = 0.30 + 0.65 * min(1.0, (t - sim.t_inspect_start) / 3.5)
        else:
            fl = max(0.25, 0.95 - 0.55 * min(1.0, (t - sim.t_solve_end) / 4.5))
        fan_load[i] = round(fl, 4)

        # Joint angular velocities from actual rig kinematics (sampled every 2nd step = 100 Hz -> smooth)
        if i % 2 == 0:
            st = sim.eval_frame(t)
            if prev_st is not None:
                dt2 = 2.0 * dt
                # Arm & wrist angular speeds
                v_arm = 0.0
                v_wri = 0.0
                v_fin = 0.0
                for side in ("R", "L"):
                    q_u0 = prev_st["arms"][side]["upper"].to_quaternion()
                    q_u1 = st["arms"][side]["upper"].to_quaternion()
                    q_f0 = prev_st["arms"][side]["fore"].to_quaternion()
                    q_f1 = st["arms"][side]["fore"].to_quaternion()
                    q_h0 = prev_st["arms"][side]["hand"].to_quaternion()
                    q_h1 = st["arms"][side]["hand"].to_quaternion()
                    v_arm += (q_u0.rotation_difference(q_u1).angle + q_f0.rotation_difference(q_f1).angle) / dt2
                    v_wri += q_h0.rotation_difference(q_h1).angle / dt2
                    for fn in FINGER_ORDER:
                        a0 = prev_st["arms"][side]["finger_angles"][fn]
                        a1 = st["arms"][side]["finger_angles"][fn]
                        v_fin += (abs(a1[0] - a0[0]) + abs(a1[1] - a0[1]) + abs(a1[2] - a0[2])) / dt2
                v_hd = (
                    prev_st["head_q"].rotation_difference(st["head_q"]).angle
                    + prev_st["neck_q"].rotation_difference(st["neck_q"]).angle
                ) / dt2
                servo_arm[i] = round(v_arm, 3)
                servo_wrist[i] = round(v_wri, 3)
                servo_finger[i] = round(v_fin, 3)
                servo_head[i] = round(v_hd, 3)
                if i > 0:
                    servo_arm[i - 1] = round(0.5 * (servo_arm[max(0, i - 2)] + servo_arm[i]), 3)
                    servo_wrist[i - 1] = round(0.5 * (servo_wrist[max(0, i - 2)] + servo_wrist[i]), 3)
                    servo_finger[i - 1] = round(0.5 * (servo_finger[max(0, i - 2)] + servo_finger[i]), 3)
                    servo_head[i - 1] = round(0.5 * (servo_head[max(0, i - 2)] + servo_head[i]), 3)
            prev_st = st

    meta = {
        "fps": sim.fps,
        "total_frames": sim.total_frames,
        "duration_s": round(sim.total_frames / sim.fps, 6),
        "sample_hz": hz,
        "scramble": sim.solve.scramble,
        "solution": sim.solve.solution,
        "qtm_count": sim.solve.qtm_count,
        "t_wake": sim.t_wake,
        "t_grasp": sim.t_grasp,
        "t_lift": sim.t_lift_start,
        "t_inspect_start": sim.t_inspect_start,
        "t_inspect_end": sim.t_inspect_end,
        "t_solve_start": sim.t_solve_start,
        "t_solve_end": sim.t_solve_end,
        "t_setdown": sim.t_setdown_touch,
        "events": sim.contact_events,
        "turns": [
            {
                "index": ev.index,
                "token": ev.token,
                "face": ev.face,
                "cw": ev.cw,
                "mode": ev.mode,
                "hand": ev.hand,
                "t_start": round(ev.t_start, 5),
                "t_capture": round(ev.t_capture, 5),
                "t_end": round(ev.t_end, 5),
                "capture_frame": int(round(ev.t_capture * sim.fps)) + 1,
            }
            for ev in sim.turns
        ],
        "shots": [
            {"name": s.name, "t_start": round(s.t_start, 4), "t_end": round(s.t_end, 4), "lens_mm": s.lens_mm}
            for s in shots
        ],
        "curves": {
            "layer_speed": layer_speed,
            "servo_arm": servo_arm,
            "servo_wrist": servo_wrist,
            "servo_finger": servo_finger,
            "servo_head": servo_head,
            "cam_dist": cam_dist,
            "cam_pan": cam_pan,
            "cam_shot": cam_shot,
            "fan_load": fan_load,
        },
    }

    if out_path is None:
        out_path = Path(__file__).resolve().parent.parent / "renders" / "sim_meta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta))
    return meta
