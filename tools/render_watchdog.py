#!/usr/bin/env python3
"""Resilient multi-worker render watchdog for long Cycles sequences.

Features:
  - Resumable: skips already complete/valid PNG frames.
  - Fault-tolerant: restarts crashed or stalled workers automatically.
  - Atomic hygiene: deletes half-written `.tmp.png` or corrupt PNGs before restart.
  - Strict isolation: uses dedicated `--frames-dir` and never touches look-dev/test renders.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from PIL import Image


def is_valid_png(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 4096:
        return False
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def clean_partials(frames_dir: Path) -> int:
    removed = 0
    if not frames_dir.exists():
        return 0
    for tmp in frames_dir.glob("*.tmp.png"):
        try:
            tmp.unlink()
            removed += 1
        except OSError:
            pass
    for png in frames_dir.glob("f*.png"):
        if not is_valid_png(png):
            try:
                png.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def missing_frames(frames_dir: Path, start: int, end: int) -> List[int]:
    return [f for f in range(start, end + 1) if not is_valid_png(frames_dir / f"f{f:04d}.png")]


def run_watchdog(
    blend: Path,
    preset: str,
    frames_dir: Path,
    start: int,
    end: int,
    workers: int = 1,
    stall_timeout_s: float = 600.0,
    use_gpu: bool = False,
):
    frames_dir.mkdir(parents=True, exist_ok=True)
    cleaned = clean_partials(frames_dir)
    if cleaned:
        print(f"[watchdog] Cleaned {cleaned} incomplete/corrupt partial PNGs.", flush=True)

    python_exe = sys.executable
    render_script = Path(__file__).resolve().parent.parent / "render.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

    total = end - start + 1
    procs: Dict[int, subprocess.Popen] = {}
    last_progress = time.monotonic()
    last_count = total - len(missing_frames(frames_dir, start, end))
    print(f"[watchdog] Starting: {last_count}/{total} frames already complete | workers={workers}, preset={preset}", flush=True)

    def spawn_worker(wid: int):
        cmd = [
            python_exe,
            str(render_script),
            "--blend", str(blend),
            "--preset", preset,
            "--anim",
            "--start", str(start + wid),
            "--end", str(end),
            "--step", str(workers),
            "--frames-dir", str(frames_dir),
        ]
        if use_gpu:
            cmd.append("--gpu")
        print(f"[watchdog] Launching worker {wid}: {' '.join(cmd)}", flush=True)
        return subprocess.Popen(cmd, env=env)

    try:
        for wid in range(workers):
            procs[wid] = spawn_worker(wid)

        while True:
            time.sleep(5.0)
            miss = missing_frames(frames_dir, start, end)
            done = total - len(miss)
            if done > last_count:
                last_count = done
                last_progress = time.monotonic()
                print(f"[watchdog] Progress: {done}/{total} frames complete ({100.0*done/total:.1f}%)", flush=True)

            if not miss:
                print(f"[watchdog] ALL {total} frames verified complete!", flush=True)
                break

            # Restart any dead worker while frames remain
            for wid, p in list(procs.items()):
                rc = p.poll()
                if rc is not None:
                    # Check if this worker's stride still has missing frames
                    stride_miss = [f for f in miss if (f - start) % workers == wid]
                    if stride_miss:
                        print(f"[watchdog] Worker {wid} exited (rc={rc}) with {len(stride_miss)} frames left — restarting.", flush=True)
                        clean_partials(frames_dir)
                        procs[wid] = spawn_worker(wid)

            # Check for global stall
            if time.monotonic() - last_progress > stall_timeout_s:
                print(f"[watchdog] STALL detected (> {stall_timeout_s:.0f}s with no new frames) — restarting all workers!", flush=True)
                for p in procs.values():
                    if p.poll() is None:
                        p.kill()
                clean_partials(frames_dir)
                last_progress = time.monotonic()
                for wid in range(workers):
                    procs[wid] = spawn_worker(wid)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()


def main():
    ap = argparse.ArgumentParser(description="Resilient Cycles render watchdog")
    ap.add_argument("--blend", default="kairo_cube.blend")
    ap.add_argument("--preset", default="final")
    ap.add_argument("--frames-dir", default="renders/frames")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=556)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--stall-timeout", type=float, default=600.0)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    run_watchdog(
        Path(args.blend).resolve(),
        args.preset,
        Path(args.frames_dir).resolve(),
        args.start,
        args.end,
        workers=args.workers,
        stall_timeout_s=args.stall_timeout,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
