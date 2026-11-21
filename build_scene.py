#!/usr/bin/env python3
"""Master scene builder: builds kairo_cube.blend + verified simulation telemetry.

Usage:
    python build_scene.py --preset final
"""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy

import kairo  # noqa: F401
from kairo import animate, cameras, cube_model, environment, geo, materials, render_setup, robot_model
from kairo.choreography import Simulation
from kairo.verify_motion import verify_simulation


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_all(preset: str = "final", use_gpu: bool = False, out_blend: Path | None = None):
    reset_scene()
    print("[1/6] Running physics simulation & per-frame verification...")
    sim = Simulation()
    rep = verify_simulation(sim, threshold_mm=0.50, verbose=True)
    if not rep["passed"]:
        raise RuntimeError(f"Motion/penetration verification failed: {rep}")

    print("[2/6] Building procedural materials & PIL micro-textures...")
    mats = materials.build_all_materials()

    print("[3/6] Building stickerless 57 mm speed cube & KAIRO-04 robot...")
    coll_cube = geo.ensure_collection("SpeedCube")
    coll_robot = geo.ensure_collection("KAIRO_04")
    coll_env = geo.ensure_collection("RoboticsLab")
    coll_cams = geo.ensure_collection("CinemaCameras")

    cube_root, cubies = cube_model.build_cube(coll_cube, mats["cube_core"], mats["cube_tiles"])
    rig = robot_model.build_robot(coll_robot, mats)

    print("[4/6] Building sunlit robotics lab environment & practical lighting...")
    environment.build_environment(coll_env, mats)

    print("[5/6] Baking physics animation & 7 cinema cameras...")
    meta = animate.bake_animation(sim, rig, cube_root, cubies)
    cameras.build_and_animate_cameras(coll_cams, sim)

    print(f"[6/6] Configuring Cycles AgX ({preset}) & saving .blend...")
    render_setup.configure_render(preset=preset, use_gpu=use_gpu, frame_end=sim.total_frames)

    if out_blend is None:
        out_blend = Path(__file__).resolve().parent / "kairo_cube.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend), compress=True)
    print(f"Saved {out_blend} ({out_blend.stat().st_size / 1e6:.2f} MB) | "
          f"{sim.total_frames} frames ({meta['duration_s']:.2f} s @ {sim.fps} fps)")
    return out_blend, meta


def main():
    ap = argparse.ArgumentParser(description="Build KAIRO-04 photorealistic speedcube scene (.blend)")
    ap.add_argument("--preset", default="final", choices=list(render_setup.PRESETS.keys()))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="kairo_cube.blend")
    args = ap.parse_args()
    build_all(preset=args.preset, use_gpu=args.gpu, out_blend=Path(args.out).resolve())


if __name__ == "__main__":
    main()
