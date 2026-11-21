# KAIRO-04 Progress Checkpoint

## Machine Measurements & Cycles Benchmarks (Stage 0)
- **Blender**: `bpy 5.0.1` (Python 3.11.2, Cycles CPU + OpenImageDenoise + AgX)
- **CPU**: 2 vCPUs (`Intel(R) Xeon(R) Platinum 8481C`)
- **RAM**: 7.8 GiB
- **GPU**: None detected (CPU-only VM; `--gpu` auto-enables OptiX/CUDA/HIP/Metal/oneAPI on GPU hosts)

### Measured Per-Frame Render Times (556 frames @ 24 fps = 23.17 s)
| Preset | Resolution | Samples | GI | Measured s/frame (2 vCPUs) | Est. 556-Frame Total (this VM) | Est. 1× RTX 4090 / A100 (`--gpu`) |
| --- | --- | --- | --- | --- | --- | --- |
| `motion` | 960×540 | 10 (adaptive) | Fast GI | `27.3 s` | `4.2 h` | `~3 min` |
| `preview` | 1280×720 | 20 (adaptive) | Fast GI | `85.3 s` | `13.2 h` | `~6 min` |
| `hd_preview` | 1920×1080 | 32 (adaptive) | Fast GI | `222.7 – 296.2 s` (avg `~259 s`) | `~40.0 h` | `~12 min` |
| `final` | 1920×1080 | 256 (adaptive) | Full GI | `> 10 min` (est. `~30–40 min`) | `~280–370 h` | `~25–40 min` |

## Completed Milestones
- [x] **Stage 0 (Machine Measured & Benchmarked)**: Full hardware + multi-preset Cycles benchmarks recorded above.
- [x] **Stage 1 (Exact 57 mm Stickerless Speedcube & Verified Kociemba Solve)**:
  - Scramble (`21 HTM`): `D L' U' F2 B' R2 L' D L2 F' D' F2 L' F L2 F' L2 D2 U2 R' B'`
  - Kociemba Solution (`20 HTM / 29 QTM`): `L2 B' D' R2 F' D R' D2 L D L F2 U D R2 L2 U' R2 F2 L2`
  - Verified move-by-move in `CubeState` from solved -> scrambled -> 100% solved.
- [x] **Stage 2 (Physics, Finger Tricks & Automated Penetration Verification)**:
  - RK2 magnetic detent flicks (`3.03°` overshoot), minimum-jerk wrist turns, 3-axis compliant grip recoil, resource-scheduler slice clearing, 9.90 s 6-face inspection, `7.08 TPS` speedsolve.
  - `kairo/verify_motion.py`: **PASS** across all `556/556` frames (`0.172 mm` max finger–cube penetration vs `0.50 mm` limit, `0.000 mm` hand–desk penetration, `0` unreachable arm IK frames).
- [x] **Stage 3 & 4 (Original KAIRO-04 Robot, Sunlit Lab & 5× 1080p Hero Stills)**:
  - Iterated look-dev across 4 visual passes addressing every detail (stickerless speedcube shells, sculpted tapered phalanges with clevis pins & silicone grip pads, ergonomic chest carapace + crisp `KAIRO-04` titanium badge, flush helmet + single-surface dark smoked obsidian visor with centered glowing iris LEDs over 100% opaque bulkhead, anchored cables, oiled American walnut desk, ESD metric mat, glowing oscilloscope & PSU rack, warm practical lamp, golden afternoon sun through mullioned windows).
  - Rendered 5 full **1920×1080** hero stills (`renders/stills/hero_1080p_*.png`) across all key beats.
- [x] **Stage 5 (Physics-Driven Sound & 20-Check Audio QA)**:
  - `renders/foley.wav` (`48 kHz / 24-bit` stereo, `23.16667 s`), **20/20 checks PASS** (`renders/foley_report.txt`), `-23.00 LUFS` confirmed independently with `ffmpeg ebur128`, `-2.68 dBTP`, `29/29` clicks within `0.33 ms` max error vs physics, `0` stray onsets, `2.4 dB` max tonal prominence.
- [x] **Stage 6 (Cloud 1080p Full 556-Frame Sequence Render & Final Verified MP4)**:
  - Rendered all `556/556` frames at `1920×1080` (`hd_preview`, 32 adaptive samples + OpenImageDenoise + AgX) across 20 parallel GitHub Actions cloud shards (`.github/workflows/render_1080p.yml`, run `44905823768`, `20/20` shards + `assemble_mp4` succeeded).
  - Deliverable: `renders/kairo_cube_1080p.mp4` (`21.8 MB`, `1920×1080` H.264 High `yuv420p`, `24 fps`, `556` frames, `23.167 s`, `7.24 Mbps` video + `48 kHz` stereo AAC-LC `265 kbps` audio).
  - Independently verified via `ffprobe`, `ffmpeg ebur128` (`-23.0 LUFS`, `-2.7 dBTP`), and 6-point visual frame extraction across the timeline (`1.5s`, `6.0s`, `10.5s`, `14.4s`, `18.75s`, `22.2s`).

