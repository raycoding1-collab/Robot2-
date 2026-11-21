#!/usr/bin/env python3
"""Resumable Cycles renderer, still look-dev tool, benchmark timer, and H.264 MP4 encoder.

Usage:
    python render.py --stills --preset lookdev              # render 4 key hero stills
    python render.py --bench --preset final --frame 338     # benchmark 1 target-quality frame
    python render.py --anim --preset motion --out renders/preview_frames
    python render.py --encode --frames-dir renders/frames --audio renders/foley.wav --mp4 renders/kairo_cube.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import bpy
from PIL import Image

import kairo  # noqa: F401
from kairo import render_setup


HERO_STILLS = [
    ("still_01_wide_f036.png",          36),   # Shot 1: sunlit lab desk establishing + reaching
    ("still_02_visor_inspect_f200.png", 200),  # Shot 3: sweeping visor + hands inspecting cube
    ("still_03_mid_solve_f346.png",     346),  # Shot 4: tight hero macro mid-flick with 180-deg shutter blur!
    ("still_04_finale_f532.png",        532),  # Shot 7: solved cube on ESD mat + KAIRO-04 looking into lens!
]


def is_valid_png(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 4096:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def render_single_frame(frame: int, out_path: Path, skip_existing: bool = True) -> float:
    """Render one frame atomically (write to .tmp.png then rename so no partial PNGs remain)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and is_valid_png(out_path):
        return 0.0

    sc = bpy.context.scene
    sc.frame_set(frame)
    # Apply timeline camera marker bound at or before `frame`
    active_cam = None
    best_fr = -10**9
    for mk in sc.timeline_markers:
        if mk.camera is not None and mk.frame <= frame and mk.frame >= best_fr:
            best_fr = mk.frame
            active_cam = mk.camera
    if active_cam is not None:
        sc.camera = active_cam

    tmp_path = out_path.with_suffix(".tmp.png")
    sc.render.filepath = str(tmp_path)
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    dt = time.perf_counter() - t0
    if not is_valid_png(tmp_path):
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Rendered frame {frame} failed PNG verification!")
    os.replace(tmp_path, out_path)
    return dt


def encode_mp4(frames_dir: Path, mp4_path: Path, audio_wav: Path | None = None, fps: int = 24, total_frames: int | None = None, crf: int = 16):
    """Verify every PNG frame is complete, enforce even H.264 dimensions, mux audio, and validate with ffprobe."""
    pngs = sorted(frames_dir.glob("f*.png"))
    if not pngs:
        raise FileNotFoundError(f"No f*.png frames found in {frames_dir}")
    if total_frames is not None and len(pngs) < total_frames:
        raise RuntimeError(f"Missing frames: found {len(pngs)} / {total_frames} in {frames_dir}")

    bad = [p.name for p in pngs if not is_valid_png(p)]
    if bad:
        raise RuntimeError(f"Corrupt/incomplete PNGs detected ({len(bad)}): {bad[:5]}")

    with Image.open(pngs[0]) as im:
        w, h = im.size
    vf_parts = []
    if w % 2 != 0 or h % 2 != 0:
        vf_parts.append(f"crop={w - (w % 2)}:{h - (h % 2)}:0:0")
    # Subtle 35 mm cinema micro-grain in final encode for organic filmic texture
    vf_parts.append("noise=alls=2:allf=t")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "f%04d.png"),
    ]
    if audio_wav is not None and audio_wav.exists():
        cmd += ["-i", str(audio_wav)]
    cmd += [
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if audio_wav is not None and audio_wav.exists():
        cmd += ["-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest"]
    cmd.append(str(mp4_path))

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    print("Encoding MP4:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Verify duration and streams via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels", "-of", "default=noprint_wrappers=1", str(mp4_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    print("=== FFPROBE VERIFICATION ===")
    print(probe.stdout.strip())


def main():
    ap = argparse.ArgumentParser(description="KAIRO-04 Cycles Renderer & Encoder")
    ap.add_argument("--blend", default="kairo_cube.blend")
    ap.add_argument("--preset", default="final", choices=list(render_setup.PRESETS.keys()))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--stills", action="store_true", help="Render the 4 key hero look-dev stills")
    ap.add_argument("--bench", action="store_true", help="Benchmark a single target frame and estimate total time")
    ap.add_argument("--frame", type=int, default=338, help="Single frame for --bench or --single")
    ap.add_argument("--single", type=str, default="", help="Render single --frame to this output path")
    ap.add_argument("--anim", action="store_true", help="Render frame range resumably")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--frames-dir", default="renders/frames")
    ap.add_argument("--encode", action="store_true", help="Verify PNGs and encode H.264 MP4")
    ap.add_argument("--audio", default="renders/foley.wav")
    ap.add_argument("--mp4", default="renders/kairo_cube.mp4")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.encode:
        wav = Path(args.audio) if args.audio else None
        encode_mp4(Path(args.frames_dir), Path(args.mp4), audio_wav=wav, crf=args.crf)
        return

    blend_path = Path(args.blend).resolve()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    render_setup.configure_render(preset=args.preset, use_gpu=args.gpu)
    sc = bpy.context.scene
    total = sc.frame_end

    if args.stills:
        out_dir = Path("renders/stills")
        for fname, fr in HERO_STILLS:
            dt = render_single_frame(fr, out_dir / fname, skip_existing=not args.overwrite)
            print(f"Rendered {fname} (f{fr}) in {dt:.1f}s", flush=True)
    elif args.bench:
        out_p = Path(f"renders/bench_{args.preset}_f{args.frame:04d}.png")
        dt = render_single_frame(args.frame, out_p, skip_existing=False)
        est_h = (dt * total) / 3600.0
        print(f"BENCHMARK [{args.preset}] frame {args.frame}: {dt:.2f} s/frame -> "
              f"{total} frames est. {est_h:.2f} hours ({dt*total/60:.1f} min) on current machine.")
    elif args.single:
        dt = render_single_frame(args.frame, Path(args.single), skip_existing=False)
        print(f"Rendered f{args.frame} -> {args.single} in {dt:.1f}s")
    elif args.anim:
        f_end = args.end if args.end > 0 else total
        fdir = Path(args.frames_dir)
        done_n = 0
        t_all = time.perf_counter()
        for f in range(args.start, f_end + 1, args.step):
            out_p = fdir / f"f{f:04d}.png"
            dt = render_single_frame(f, out_p, skip_existing=not args.overwrite)
            if dt > 0:
                done_n += 1
                if done_n % 5 == 0 or f == f_end:
                    print(f"[f{f:04d}/{f_end:04d}] {dt:.1f}s (elapsed {(time.perf_counter()-t_all)/60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
