#!/usr/bin/env python3
"""Synthesize renders/foley.wav directly from renders/sim_meta.json."""
from __future__ import annotations

import json
from pathlib import Path

from kairo.sound import synthesize_soundtrack


def main():
    root = Path(__file__).resolve().parent
    meta_path = root / "renders" / "sim_meta.json"
    wav_path = root / "renders" / "foley.wav"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing — run build_scene.py first!")
    info = synthesize_soundtrack(meta_path, wav_path)
    (root / "renders" / "foley_stems.json").write_text(json.dumps(info, indent=2))
    print(f"Synthesized {wav_path} ({wav_path.stat().st_size / 1e6:.2f} MB) | "
          f"LUFS={info['lufs']:.2f}, TP={info['true_peak_dbtp']:.2f} dBTP")


if __name__ == "__main__":
    main()
