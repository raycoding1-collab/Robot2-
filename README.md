# KAIRO-04 — Photorealistic Humanoid Speedcubing Short Film

A 100% procedural, physically simulated Blender (Cycles / AgX) short film built purely from Python with zero downloaded assets. An original humanoid laboratory dexterity robot (**KAIRO-04**) sits at a sunlit robotics workbench, picks up a scrambled 57 mm modern stickerless speed cube, carefully inspects all six faces over 9.9 seconds, solves it at 7.08 quarter-turns per second using real finger tricks and magnetic detent dynamics, presents the solved cube, sets it down, and meets the camera lens.

## Repository Layout

```
build_scene.py          -> builds kairo_cube.blend + renders/sim_meta.json (verified physics)
render.py               -> resumable Cycles renderer (stills / bench / anim / H.264 MP4 mux)
make_sound.py           -> synthesizes renders/foley.wav from renders/sim_meta.json
tools/tune_grips.py     -> global differential-evolution optimizer for zero-penetration grips
tools/verify_sound.py   -> 20-check objective audio QA + independent ffmpeg EBU R128 check
tools/render_watchdog.py -> crash/stall-resilient resumable multi-worker render watchdog
kairo/                  -> modular Python package (solver, physics, IK, rig, shaders, lab, sound)
renders/                -> hero stills, audio WAV & QA report/plots, preview/final MP4
```

## Simulation & Craft Highlights

| Subsystem | Implementation |
| --- | --- |
| **Cube & Verified Solve** | Exact 26-cubie 57.0 mm stickerless speed cube (`19.0 mm` pitch, rounded wrap-around colored ABS shells, Florian corner-cutting gaps revealing internal core & stalks, UV-coat smudges & micro-scratches). Legal 21-move scramble (`D L' U' F2 B' R2 L' D L2 F' D' F2 L' F L2 F' L2 D2 U2 R' B'`) solved via Kociemba two-phase in 20 HTM / 29 QTM (`L2 B' D' R2 F' D R' D2 L D L F2 U D R2 L2 U' R2 F2 L2`) and verified step-by-step to 100% solved state before baking. |
| **Physics & Finger Tricks** | RK2-integrated magnetic speedcube flick dynamics (sinusoidal finger push -> free coast with Coulomb + viscous friction -> nonlinear neodymium detent capture with `3.03 deg` authentic overshoot & settle), minimum-jerk wrist turns for R/L, 3-axis compliant grip recoil torque, real finger tricks (`U/U'` index, `D/D'` ring, `F/F'` index with wrist-pop, `B/B'` index, `R/L` wrist turns), resource-scheduler slice clearing, and lifted clearance arcs around the cube. |
| **Automated Penetration QA** | `kairo/verify_motion.py` checks every phalanx capsule and palm hull against exact animated spinning-layer OBBs and desk height on every frame (`556/556` PASS: `0.172 mm` max finger–cube penetration vs `0.50 mm` limit, `0.000 mm` hand–desk penetration, `0` unreachable arm IK frames). |
| **Original Robot Design** | **KAIRO-04**: warm satin white ceramic multi-panel shells with subtle per-zone tone variation, orange-peel clearcoat bump, AO crevice grime, anodized graphite structural core, CNC brushed titanium harmonic-drive actuators, M2.5/M4 hex socket bolts, laser-etched titanium serial badge, properly anchored flex cables, sculpted tapered fingers with recessed clevis pins & grooved matte silicone pads, and a wide compound-curved dark smoked optical visor backed by a 100% opaque optical bulkhead housing articulated optic iris lights. |
| **Photorealistic Lab & Camera** | Cycles with AgX (`Medium High Contrast`), physical `MULTIPLE_SCATTERING` sky & golden late-afternoon sun through tall mullioned windows (unobstructed exterior view), warm practical bench lamp, overhead linear LED, invisible glossy-free bounce fill, oiled American walnut workbench, metric ESD mat, lived-in props, vertical acoustic slats, and rear equipment rack with glowing oscilloscope & PSU. Filmed across 7 physical cinema cameras (34–95 mm, 7-blade bokeh, rack focus pulls, organic handheld drift, 180° shutter, bloom, lens distortion, CA, vignette, film grain). |
| **Physics-Driven Sound** | `make_sound.py` synthesizes 48 kHz / 24-bit stereo foley directly from `renders/sim_meta.json`: multi-mode hollow-ABS modal detent clicks (`max |0.33| ms` vs physics!), angular-speed slide friction, silicone fingertip taps, desk landing thud, real joint-velocity harmonic-drive servo whir, relay click, cooling fan & studio room tone through camera-following stereo perspective and synthesized lab convolution reverb (`RT60 ≈ 0.46 s`). Mastered to EBU R128 (`-23.00 LUFS`, true peak `-2.8 dBTP`), verified 20/20 checks including `ffmpeg ebur128`. |

## Running the Pipeline

Requires Python 3.11+ with `bpy==5.0.1`, `numpy`, `scipy`, `kociemba`, `pillow`, `matplotlib`, and `ffmpeg`/`ffprobe`:

```bash
python build_scene.py --preset final             # builds kairo_cube.blend + renders/sim_meta.json
python make_sound.py                             # synthesizes renders/foley.wav (48 kHz / 24-bit)
python tools/verify_sound.py                     # runs 20-check audio QA + ffmpeg EBU R128 check
python render.py --stills --preset hero          # renders the 4 hero 1080p stills
python render.py --bench --preset final --frame 346
python tools/render_watchdog.py --preset final --workers 1 --gpu   # resumable full render
python render.py --encode --frames-dir renders/frames --audio renders/foley.wav --mp4 renders/kairo_cube.mp4
```

## Render Presets & Measured Speed (`kairo/render_setup.py`)

Measured on a **2-vCPU Xeon Platinum 8481C (no GPU)** with Blender `bpy 5.0.1` + OpenImageDenoise (556 frames = 23.17 s @ 24 fps):

| Preset | Resolution | Samples | GI | Measured (2 vCPUs) | Est. 556 Frames (2 vCPUs) | Est. 1× RTX 4090/A100 (`--gpu`) |
| --- | --- | --- | --- | --- | --- | --- |
| `motion` | 960×540 | 10 (adaptive) | Fast GI | `27.3 s/frame` | `4.2 h` | `~3 min` |
| `preview` | 1280×720 | 20 (adaptive) | Fast GI | `85.3 s/frame` | `13.2 h` | `~6 min` |
| `hd_preview` | 1920×1080 | 32 (adaptive) | Fast GI | `223 – 296 s/frame` | `~40 h` | `~12 min` |
| `hero` | 1920×1080 | 128 (adaptive) | Full GI | `> 10 min/frame` | `~140–180 h` | `~18–25 min` |
| `final` | 1920×1080 | 256 (adaptive) | Full GI | `est. 30–40 min/frame` | `~280–370 h` | `~25–40 min` |
| `cinema4k` | 3840×2160 | 512 (adaptive) | Full GI | — | — | `~2–3 h` |

### Splitting Across Multiple Machines

`render.py` skips completed valid PNGs atomically and accepts `--start`, `--end`, and `--step`, so you can split the 556 frames cleanly across `N` machines (e.g., 4 machines with `--start 1..4 --step 4`) and then encode once all `f0001.png .. f0556.png` are gathered.
