"""Photorealistic PBR material library + procedural PIL surface textures for KAIRO-04 & lab.

Zero external assets: all decals, ESD mat grids, oscilloscope readouts, and surface
imperfections are generated deterministically from Python and packed into the .blend file.
Every surface has physical breakup: AO crevice grime, multi-scale roughness variation,
orange-peel clearcoat bump, micro-scratches, and subtle fingerprint smudges on gloss.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import bpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TEX_DIR = Path(__file__).resolve().parent.parent / "renders" / "_tex"


def _load_packed_image(filepath: Path, colorspace: str = "sRGB") -> bpy.types.Image:
    img = bpy.data.images.load(str(filepath), check_existing=True)
    img.colorspace_settings.name = colorspace
    try:
        img.pack()
    except Exception:
        pass
    return img


def generate_textures() -> Dict[str, Path]:
    """Generate crisp procedural 2D graphics (ESD mat grid, KAIRO-04 badge, lab scope display)."""
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # 1. Precision ESD anti-static cutting/workbench mat (2048 x 1024)
    esd_path = TEX_DIR / "esd_mat.png"
    W, H = 2048, 1024
    rng = np.random.default_rng(42)
    # Dark slate-teal matte rubber with subtle worn central work patch
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W * 0.50, H * 0.52
    wear = np.exp(-(((xx - cx) / 420.0) ** 2 + ((yy - cy) / 240.0) ** 2))
    grain = rng.normal(0.0, 2.2, (H, W))
    base_r = np.clip(26 + 5 * wear + grain, 12, 48).astype(np.uint8)
    base_g = np.clip(33 + 6 * wear + grain, 16, 56).astype(np.uint8)
    base_b = np.clip(36 + 5 * wear + grain, 18, 60).astype(np.uint8)
    arr = np.stack([base_r, base_g, base_b], axis=-1)
    im = Image.fromarray(arr, mode="RGB")
    dr = ImageDraw.Draw(im)

    # Outer border & metric 10 mm / 50 mm coordinate grid (mat is 760 mm x 420 mm -> ~2.6 px/mm)
    pad = 64
    dr.rectangle([pad, pad, W - pad, H - pad], outline=(82, 104, 108), width=3)
    step_minor = 48
    step_major = step_minor * 5
    for x in range(pad, W - pad + 1, step_minor):
        is_maj = ((x - pad) % step_major) == 0
        col = (66, 84, 88) if is_maj else (42, 53, 56)
        dr.line([(x, pad), (x, H - pad)], fill=col, width=2 if is_maj else 1)
    for y in range(pad, H - pad + 1, step_minor):
        is_maj = ((y - pad) % step_major) == 0
        col = (66, 84, 88) if is_maj else (42, 53, 56)
        dr.line([(pad, y), (W - pad, y)], fill=col, width=2 if is_maj else 1)

    # Subtle target crosshairs at center cube zone
    dr.ellipse([int(cx - 95), int(cy - 95), int(cx + 95), int(cy + 95)], outline=(80, 102, 106), width=2)
    dr.line([(cx - 125, cy), (cx + 125, cy)], fill=(86, 110, 114), width=2)
    dr.line([(cx, cy - 125), (cx, cy + 125)], fill=(86, 110, 114), width=2)

    # Corner technical silkscreen markings
    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 26)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
    except Exception:
        font_lg = ImageFont.load_default()
        font_sm = font_lg
    dr.text((pad + 18, pad + 14), "ESD-SAFE WORKSTATION // BAY 04", fill=(112, 138, 142), font=font_lg)
    dr.text((W - pad - 340, H - pad - 36), "METRIC 10mm  |  R < 10^8 OHM", fill=(100, 124, 128), font=font_sm)
    # Fine millimeter ruler ticks along bottom edge
    for i, x in enumerate(range(pad, W - pad + 1, 12)):
        th = 18 if (i % 4 == 0) else 9
        dr.line([(x, H - pad), (x, H - pad - th)], fill=(96, 118, 122), width=1)
    im.save(esd_path)
    paths["esd_mat"] = esd_path

    # 2. Oscilloscope / lab telemetry display (1024 x 640) for layered background realism
    scope_path = TEX_DIR / "scope_screen.png"
    SW, SH = 1024, 640
    im_s = Image.new("RGB", (SW, SH), (10, 14, 18))
    ds = ImageDraw.Draw(im_s)
    # Graticule grid
    for x in range(64, SW - 64, 64):
        ds.line([(x, 48), (x, SH - 48)], fill=(22, 34, 42), width=1)
    for y in range(48, SH - 48, 54):
        ds.line([(64, y), (SW - 64, y)], fill=(22, 34, 42), width=1)
    ds.rectangle([64, 48, SW - 64, SH - 48], outline=(48, 72, 84), width=2)
    # Multi-channel torque & angular velocity traces (cyan & warm amber)
    pts1, pts2 = [], []
    for x in range(68, SW - 68, 4):
        u = (x - 68) / (SW - 136)
        # Damped magnetic-detent step response trace!
        y1 = 210 - int(85 * math.exp(-3.2 * ((u * 4.0) % 1.0)) * math.sin(18.0 * u))
        y2 = 420 - int(65 * math.sin(2.0 * math.pi * 3.0 * u) * math.cos(7.5 * u))
        pts1.append((x, y1))
        pts2.append((x, y2))
    ds.line(pts1, fill=(75, 225, 245), width=4)
    ds.line(pts2, fill=(250, 175, 65), width=4)
    ds.text((76, 16), "CH1: TORQUE 1.42 Nm   CH2: DETENT VEL 28.4 rad/s   TRIG: AUTO", fill=(165, 215, 225), font=font_sm)
    im_s.save(scope_path)
    paths["scope_screen"] = scope_path

    # 3. KAIRO-04 laser-etched titanium serial badge (1024 x 384)
    badge_path = TEX_DIR / "kairo_badge.png"
    BW, BH = 1024, 384
    im_b = Image.new("RGB", (BW, BH), (128, 132, 138))
    db = ImageDraw.Draw(im_b)
    db.rectangle([16, 16, BW - 16, BH - 16], outline=(48, 52, 58), width=5)
    db.rectangle([28, 28, 58, BH - 28], fill=(212, 118, 42))
    try:
        f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 112)
        f_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 40)
    except Exception:
        f_title = f_sub = ImageFont.load_default()
    db.text((92, 62), "KAIRO-04", fill=(28, 30, 34), font=f_title)
    db.text((96, 218), "DEXTERITY PLATFORM // UNIT 04B", fill=(42, 46, 52), font=f_sub)
    im_b.save(badge_path)
    paths["kairo_badge"] = badge_path

    return paths


# ==============================================================================
# Node-tree helpers
# ==============================================================================
def _new_mat(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf


def _tex_coord(nt, x=-950, y=0):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (x, y)
    return tc


def _add_crevice_grime(nt, bsdf, base_rgb, dirt_rgb=(0.14, 0.135, 0.125), ao_dist=0.022, strength=0.38):
    """Mix subtle physical AO crevice dirt into Base Color so panel gaps & bolts look lived-in."""
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 2
    ao.inputs["Distance"].default_value = ao_dist
    ao.location = (-360, 260)

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-160, 260)
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 0.92
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(ao.outputs["AO"], ramp.inputs["Fac"])

    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.location = (40, 260)
    mix.inputs["A"].default_value = (*dirt_rgb, 1.0)
    mix.inputs["B"].default_value = (*base_rgb, 1.0)
    # Blend factor: 1.0 - strength * (1 - AO) so open surfaces keep pure base color
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "MULTIPLY_ADD"
    inv.location = (-20, 400)
    inv.inputs[1].default_value = strength
    inv.inputs[2].default_value = 1.0 - strength
    nt.links.new(ramp.outputs["Color"], inv.inputs[0])
    nt.links.new(inv.outputs["Value"], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    return ramp


def _make_ceramic_mat(name: str, base_rgb, rough_lo=0.14, rough_hi=0.29, seed_off=0.0):
    """Satin industrial white ceramic shell with orange-peel clearcoat, AO grime, smudges & micro-scratches."""
    mat, nt, bsdf = _new_mat(name)
    tc = _tex_coord(nt)

    # Subtle per-part macro color drift + AO crevice dirt
    ao_ramp = _add_crevice_grime(nt, bsdf, base_rgb, dirt_rgb=(0.18, 0.175, 0.165), ao_dist=0.024, strength=0.42)

    # Multi-scale roughness: macro handling smudges + micro ceramic grain + crevice roughness bump
    n_smudge = nt.nodes.new("ShaderNodeTexNoise")
    n_smudge.location = (-680, -40)
    n_smudge.inputs["Scale"].default_value = 14.0
    n_smudge.inputs["Detail"].default_value = 6.0
    n_smudge.inputs["Roughness"].default_value = 0.58
    n_smudge.inputs["Distortion"].default_value = 0.45
    nt.links.new(tc.outputs["Object"], n_smudge.inputs["Vector"])

    n_micro = nt.nodes.new("ShaderNodeTexNoise")
    n_micro.location = (-680, -240)
    n_micro.inputs["Scale"].default_value = 180.0
    n_micro.inputs["Detail"].default_value = 8.0
    n_micro.inputs["Roughness"].default_value = 0.65
    nt.links.new(tc.outputs["Object"], n_micro.inputs["Vector"])

    # Combine macro smudges & fine micro-grain
    m_rough = nt.nodes.new("ShaderNodeMix")
    m_rough.data_type = "FLOAT"
    m_rough.location = (-400, -100)
    m_rough.inputs["Factor"].default_value = 0.35
    nt.links.new(n_smudge.outputs["Fac"], m_rough.inputs["A"])
    nt.links.new(n_micro.outputs["Fac"], m_rough.inputs["B"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = (-180, -100)
    mr.inputs["From Min"].default_value = 0.28
    mr.inputs["From Max"].default_value = 0.72
    mr.inputs["To Min"].default_value = rough_lo
    mr.inputs["To Max"].default_value = rough_hi
    nt.links.new(m_rough.outputs["Result"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    # Fine surface bump + subtle micro-scratches
    bmp_base = nt.nodes.new("ShaderNodeBump")
    bmp_base.location = (20, -260)
    bmp_base.inputs["Strength"].default_value = 0.025
    bmp_base.inputs["Distance"].default_value = 0.0004
    nt.links.new(n_micro.outputs["Fac"], bmp_base.inputs["Height"])
    nt.links.new(bmp_base.outputs["Normal"], bsdf.inputs["Normal"])

    # Authentic orange-peel clearcoat (catches glancing window light like real painted ceramic!)
    bsdf.inputs["Coat Weight"].default_value = 0.72
    bsdf.inputs["Coat Roughness"].default_value = 0.085
    bsdf.inputs["IOR"].default_value = 1.52
    bsdf.inputs["Specular IOR Level"].default_value = 0.55

    n_peel = nt.nodes.new("ShaderNodeTexNoise")
    n_peel.location = (-440, -440)
    n_peel.inputs["Scale"].default_value = 340.0
    n_peel.inputs["Detail"].default_value = 4.0
    n_peel.inputs["Roughness"].default_value = 0.45
    nt.links.new(tc.outputs["Object"], n_peel.inputs["Vector"])

    bmp_coat = nt.nodes.new("ShaderNodeBump")
    bmp_coat.location = (20, -440)
    bmp_coat.inputs["Strength"].default_value = 0.042
    bmp_coat.inputs["Distance"].default_value = 0.0003
    nt.links.new(n_peel.outputs["Fac"], bmp_coat.inputs["Height"])
    nt.links.new(bmp_coat.outputs["Normal"], bsdf.inputs["Coat Normal"])

    return mat


def _make_metal_mat(name: str, base_rgb, rough_lo=0.22, rough_hi=0.38, metallic=0.92, aniso=0.55, brush_scale=(18, 220, 18)):
    """CNC-machined anodized graphite or brushed titanium with directional streaks, wear & AO."""
    mat, nt, bsdf = _new_mat(name)
    tc = _tex_coord(nt)
    _add_crevice_grime(nt, bsdf, base_rgb, dirt_rgb=(0.015, 0.016, 0.018), ao_dist=0.018, strength=0.45)

    mp = nt.nodes.new("ShaderNodeMapping")
    mp.location = (-740, -140)
    mp.inputs["Scale"].default_value = brush_scale
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    n_br = nt.nodes.new("ShaderNodeTexNoise")
    n_br.location = (-520, -140)
    n_br.inputs["Scale"].default_value = 1.0
    n_br.inputs["Detail"].default_value = 8.0
    n_br.inputs["Roughness"].default_value = 0.62
    nt.links.new(mp.outputs["Vector"], n_br.inputs["Vector"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = (-260, -140)
    mr.inputs["From Min"].default_value = 0.30
    mr.inputs["From Max"].default_value = 0.70
    mr.inputs["To Min"].default_value = rough_lo
    mr.inputs["To Max"].default_value = rough_hi
    nt.links.new(n_br.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.location = (0, -240)
    bmp.inputs["Strength"].default_value = 0.045
    bmp.inputs["Distance"].default_value = 0.00025
    nt.links.new(n_br.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Metallic"].default_value = metallic
    if "Anisotropic" in bsdf.inputs:
        bsdf.inputs["Anisotropic"].default_value = aniso
    return mat


def _make_silicone_mat(name: str = "KAIRO_Silicone"):
    """Tactile dark matte elastomer fingertip & palmar grip pads with micro-pebble grain."""
    mat, nt, bsdf = _new_mat(name)
    tc = _tex_coord(nt)
    _add_crevice_grime(nt, bsdf, (0.028, 0.030, 0.034), dirt_rgb=(0.010, 0.010, 0.012), ao_dist=0.012, strength=0.35)

    n_peb = nt.nodes.new("ShaderNodeTexNoise")
    n_peb.location = (-520, -120)
    n_peb.inputs["Scale"].default_value = 650.0
    n_peb.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], n_peb.inputs["Vector"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = (-260, -80)
    mr.inputs["From Min"].default_value = 0.3
    mr.inputs["From Max"].default_value = 0.7
    mr.inputs["To Min"].default_value = 0.56
    mr.inputs["To Max"].default_value = 0.76
    nt.links.new(n_peb.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.location = (0, -220)
    bmp.inputs["Strength"].default_value = 0.14
    bmp.inputs["Distance"].default_value = 0.00035
    nt.links.new(n_peb.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    return mat


def _make_visor_mat(name: str = "KAIRO_VisorGlass"):
    """Dark smoked optical polycarbonate visor with faint fingerprint smudges & high-gloss reflection."""
    mat, nt, bsdf = _new_mat(name)
    tc = _tex_coord(nt)
    bsdf.inputs["Base Color"].default_value = (0.012, 0.014, 0.018, 1.0)
    bsdf.inputs["IOR"].default_value = 1.54
    bsdf.inputs["Transmission Weight"].default_value = 0.48
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat IOR"].default_value = 1.56

    # Faint oil/smudge roughness patches visible in window reflections
    n_sm = nt.nodes.new("ShaderNodeTexNoise")
    n_sm.location = (-520, -80)
    n_sm.inputs["Scale"].default_value = 26.0
    n_sm.inputs["Detail"].default_value = 5.0
    n_sm.inputs["Distortion"].default_value = 0.75
    nt.links.new(tc.outputs["Object"], n_sm.inputs["Vector"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = (-240, -80)
    mr.inputs["From Min"].default_value = 0.32
    mr.inputs["From Max"].default_value = 0.72
    mr.inputs["To Min"].default_value = 0.016
    mr.inputs["To Max"].default_value = 0.068
    nt.links.new(n_sm.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    mr_c = nt.nodes.new("ShaderNodeMapRange")
    mr_c.location = (-240, -280)
    mr_c.inputs["From Min"].default_value = 0.38
    mr_c.inputs["From Max"].default_value = 0.76
    mr_c.inputs["To Min"].default_value = 0.010
    mr_c.inputs["To Max"].default_value = 0.055
    nt.links.new(n_sm.outputs["Fac"], mr_c.inputs["Value"])
    nt.links.new(mr_c.outputs["Result"], bsdf.inputs["Coat Roughness"])
    return mat


def _make_emission_mat(name: str, rgb, strength: float):
    mat, nt, bsdf = _new_mat(name)
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Emission Strength"].default_value = strength
    bsdf.inputs["Roughness"].default_value = 0.2
    return mat


# Calibrated modern stickerless speedcube ABS colors (linear sRGB under AgX)
CUBE_PALETTE = {
    "white":  (0.84, 0.83, 0.80),
    "yellow": (0.86, 0.68, 0.018),
    "red":    (0.62, 0.022, 0.018),
    "orange": (0.86, 0.16, 0.008),
    "green":  (0.018, 0.44, 0.065),
    "blue":   (0.012, 0.11, 0.56),
}


def _make_cube_tile_mat(color_name: str):
    """UV-coated stickerless speedcube ABS with subtle SSS, oil smudges & wipe micro-scratches."""
    mat, nt, bsdf = _new_mat(f"CubeABS_{color_name}")
    tc = _tex_coord(nt)
    rgb = CUBE_PALETTE[color_name]

    # Per-cubie microscopic tone variation + subtle AO seam darkening
    obj_info = nt.nodes.new("ShaderNodeObjectInfo")
    obj_info.location = (-720, 340)
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.location = (-440, 300)
    hsv.inputs["Color"].default_value = (*rgb, 1.0)
    mr_v = nt.nodes.new("ShaderNodeMapRange")
    mr_v.location = (-580, 340)
    mr_v.inputs["To Min"].default_value = 0.96
    mr_v.inputs["To Max"].default_value = 1.04
    nt.links.new(obj_info.outputs["Random"], mr_v.inputs["Value"])
    nt.links.new(mr_v.outputs["Result"], hsv.inputs["Value"])

    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 2
    ao.inputs["Distance"].default_value = 0.006
    ao.location = (-440, 140)
    mix_ao = nt.nodes.new("ShaderNodeMix")
    mix_ao.data_type = "RGBA"
    mix_ao.blend_type = "MULTIPLY"
    mix_ao.inputs["Factor"].default_value = 0.32
    mix_ao.location = (-160, 220)
    nt.links.new(hsv.outputs["Color"], mix_ao.inputs["A"])
    nt.links.new(ao.outputs["Color"], mix_ao.inputs["B"])
    nt.links.new(mix_ao.outputs["Result"], bsdf.inputs["Base Color"])

    # Real speedcube glossy UV coat + fingerprint smudge breakup + fine wipe scratches!
    n_smudge = nt.nodes.new("ShaderNodeTexNoise")
    n_smudge.location = (-680, -60)
    n_smudge.inputs["Scale"].default_value = 65.0
    n_smudge.inputs["Detail"].default_value = 5.0
    n_smudge.inputs["Distortion"].default_value = 0.95
    nt.links.new(tc.outputs["Object"], n_smudge.inputs["Vector"])

    # Fine micro-scratches from thousands of speedcube turns
    mp_sc = nt.nodes.new("ShaderNodeMapping")
    mp_sc.location = (-860, -280)
    mp_sc.inputs["Scale"].default_value = (140.0, 28.0, 140.0)
    nt.links.new(tc.outputs["Object"], mp_sc.inputs["Vector"])
    n_sc = nt.nodes.new("ShaderNodeTexNoise")
    n_sc.location = (-640, -280)
    n_sc.inputs["Scale"].default_value = 2.4
    n_sc.inputs["Detail"].default_value = 10.0
    n_sc.inputs["Roughness"].default_value = 0.72
    nt.links.new(mp_sc.outputs["Vector"], n_sc.inputs["Vector"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = (-280, -60)
    mr.inputs["From Min"].default_value = 0.30
    mr.inputs["From Max"].default_value = 0.72
    mr.inputs["To Min"].default_value = 0.055
    mr.inputs["To Max"].default_value = 0.165
    nt.links.new(n_smudge.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.location = (0, -260)
    bmp.inputs["Strength"].default_value = 0.022
    bmp.inputs["Distance"].default_value = 0.00018
    nt.links.new(n_sc.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Coat Normal"])

    bsdf.inputs["IOR"].default_value = 1.53
    bsdf.inputs["Coat Weight"].default_value = 0.88
    bsdf.inputs["Coat Roughness"].default_value = 0.045
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.045
    if "Subsurface Scale" in bsdf.inputs:
        bsdf.inputs["Subsurface Scale"].default_value = 0.0022
    return mat


def _make_wood_mat(name: str = "Lab_WalnutDesk", vertical: bool = False):
    """Authentic oiled American walnut with fine longitudinal wood fibers, satin sheen & pore bump."""
    mat, nt, bsdf = _new_mat(name)
    tc = _tex_coord(nt)
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.location = (-820, 120)
    # Long straight fibers along X for tabletop, or along Z for vertical acoustic wall slats!
    mp.inputs["Scale"].default_value = (14.0, 14.0, 0.65) if vertical else (0.75, 16.0, 6.0)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    # Gentle organic cathedral rings (low distortion -> never zebra/camo!)
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X" if vertical else "Y"
    wave.inputs["Scale"].default_value = 1.35
    wave.inputs["Distortion"].default_value = 1.65
    wave.inputs["Detail"].default_value = 5.0
    wave.inputs["Detail Scale"].default_value = 1.4
    wave.location = (-580, 180)
    nt.links.new(mp.outputs["Vector"], wave.inputs["Vector"])

    # Fine fibrous micro-grain streaks
    n_fib = nt.nodes.new("ShaderNodeTexNoise")
    n_fib.location = (-580, -40)
    n_fib.inputs["Scale"].default_value = 12.0
    n_fib.inputs["Detail"].default_value = 10.0
    n_fib.inputs["Roughness"].default_value = 0.72
    nt.links.new(mp.outputs["Vector"], n_fib.inputs["Vector"])

    mix_g = nt.nodes.new("ShaderNodeMix")
    mix_g.data_type = "FLOAT"
    mix_g.location = (-380, 120)
    mix_g.inputs["Factor"].default_value = 0.45
    nt.links.new(wave.outputs["Fac"], mix_g.inputs["A"])
    nt.links.new(n_fib.outputs["Fac"], mix_g.inputs["B"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-200, 160)
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (0.036, 0.017, 0.008, 1.0)
    e_mid = ramp.color_ramp.elements.new(0.56)
    e_mid.color = (0.078, 0.038, 0.018, 1.0)
    ramp.color_ramp.elements[2].position = 0.90
    ramp.color_ramp.elements[2].color = (0.125, 0.064, 0.031, 1.0)
    nt.links.new(mix_g.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = (-200, -80)
    mr.inputs["To Min"].default_value = 0.28
    mr.inputs["To Max"].default_value = 0.44
    nt.links.new(n_fib.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.location = (20, -220)
    bmp.inputs["Strength"].default_value = 0.035
    bmp.inputs["Distance"].default_value = 0.0004
    nt.links.new(n_fib.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Coat Weight"].default_value = 0.26
    bsdf.inputs["Coat Roughness"].default_value = 0.26
    return mat


def _make_image_mat(name: str, img_path: Path, rough_lo=0.55, rough_hi=0.72, emission_str=0.0):
    mat, nt, bsdf = _new_mat(name)
    tc = _tex_coord(nt)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.location = (-360, 140)
    tex.image = _load_packed_image(img_path, "sRGB")
    nt.links.new(tc.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    if emission_str > 0.0:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = emission_str
        bsdf.inputs["Roughness"].default_value = 0.08
        bsdf.inputs["Coat Weight"].default_value = 0.8
        return mat

    n = nt.nodes.new("ShaderNodeTexNoise")
    n.location = (-500, -120)
    n.inputs["Scale"].default_value = 45.0
    n.inputs["Detail"].default_value = 5.0
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.location = (-220, -120)
    mr.inputs["To Min"].default_value = rough_lo
    mr.inputs["To Max"].default_value = rough_hi
    nt.links.new(n.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.location = (0, -240)
    bmp.inputs["Strength"].default_value = 0.04
    bmp.inputs["Distance"].default_value = 0.0004
    nt.links.new(n.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_all_materials() -> Dict[str, object]:
    """Build and return the complete dictionary of robot, cube, and lab environment materials."""
    tex_paths = generate_textures()
    mats: Dict[str, object] = {}

    # Robot ceramic shells — subtle per-assembly tone & roughness variation
    mats["ceramic_head"] = _make_ceramic_mat("KAIRO_CeramicHead", (0.78, 0.77, 0.735), 0.13, 0.25)
    mats["ceramic_body"] = _make_ceramic_mat("KAIRO_CeramicBody", (0.75, 0.74, 0.710), 0.16, 0.29)
    mats["ceramic_arm"] = _make_ceramic_mat("KAIRO_CeramicArm", (0.73, 0.72, 0.690), 0.16, 0.30)
    mats["ceramic_hand"] = _make_ceramic_mat("KAIRO_CeramicHand", (0.77, 0.76, 0.725), 0.14, 0.27)

    # Robot structural metals, rubber & visor
    mats["graphite"] = _make_metal_mat("KAIRO_Graphite", (0.048, 0.052, 0.058), 0.26, 0.44, metallic=0.82, aniso=0.45)
    mats["titanium"] = _make_metal_mat("KAIRO_Titanium", (0.34, 0.33, 0.31), 0.18, 0.34, metallic=0.96, aniso=0.68)
    mats["silicone"] = _make_silicone_mat("KAIRO_Silicone")
    mats["cable"] = _make_silicone_mat("KAIRO_FlexCable")
    mats["visor_glass"] = _make_visor_mat("KAIRO_VisorGlass")

    # 100% opaque matte optical bulkhead behind visor
    blk, _, b_bsdf = _new_mat("KAIRO_OpticalBlack")
    b_bsdf.inputs["Base Color"].default_value = (0.003, 0.003, 0.004, 1.0)
    b_bsdf.inputs["Roughness"].default_value = 0.94
    b_bsdf.inputs["Specular IOR Level"].default_value = 0.05
    mats["optical_black"] = blk

    # Subtle warm-ice articulated eye optics & chest telemetry LED
    mats["eye_led"] = _make_emission_mat("KAIRO_EyeOpticLED", (0.78, 0.91, 1.0), strength=8.5)
    mats["status_led"] = _make_emission_mat("KAIRO_StatusLED", (1.0, 0.78, 0.42), strength=3.2)
    mats["badge"] = _make_image_mat("KAIRO_SerialBadge", tex_paths["kairo_badge"], rough_lo=0.22, rough_hi=0.36)

    # Stickerless speedcube materials
    core, _, c_bsdf = _new_mat("CubeCore_MatteABS")
    c_bsdf.inputs["Base Color"].default_value = (0.022, 0.023, 0.026, 1.0)
    c_bsdf.inputs["Roughness"].default_value = 0.52
    mats["cube_core"] = core
    mats["cube_tiles"] = {c: _make_cube_tile_mat(c) for c in CUBE_PALETTE}

    # Lab environment materials
    mats["walnut"] = _make_wood_mat("Lab_WalnutDesk")
    mats["esd_mat"] = _make_image_mat("Lab_ESDMat", tex_paths["esd_mat"], rough_lo=0.58, rough_hi=0.76)
    mats["scope_screen"] = _make_image_mat("Lab_ScopeScreen", tex_paths["scope_screen"], emission_str=2.6)
    mats["aluminum"] = _make_metal_mat("Lab_BrushedAlum", (0.58, 0.59, 0.61), 0.16, 0.30, metallic=0.95, aniso=0.72)
    mats["dark_steel"] = _make_metal_mat("Lab_DarkSteel", (0.055, 0.058, 0.064), 0.32, 0.48, metallic=0.86, aniso=0.35)
    mats["brass"] = _make_metal_mat("Lab_Brass", (0.64, 0.46, 0.16), 0.18, 0.32, metallic=0.96, aniso=0.55)

    # Architectural plaster / concrete wall (subtle warm-neutral texture)
    wall, wnt, wbsdf = _new_mat("Lab_ConcreteWall")
    wtc = _tex_coord(wnt)
    wn = wnt.nodes.new("ShaderNodeTexNoise")
    wn.inputs["Scale"].default_value = 6.5
    wn.inputs["Detail"].default_value = 8.0
    wn.inputs["Roughness"].default_value = 0.68
    wnt.links.new(wtc.outputs["Object"], wn.inputs["Vector"])
    wr = wnt.nodes.new("ShaderNodeValToRGB")
    wr.color_ramp.elements[0].color = (0.24, 0.235, 0.225, 1.0)
    wr.color_ramp.elements[1].color = (0.34, 0.33, 0.315, 1.0)
    wnt.links.new(wn.outputs["Fac"], wr.inputs["Fac"])
    wnt.links.new(wr.outputs["Color"], wbsdf.inputs["Base Color"])
    wbsdf.inputs["Roughness"].default_value = 0.78
    wbmp = wnt.nodes.new("ShaderNodeBump")
    wbmp.inputs["Strength"].default_value = 0.06
    wbmp.inputs["Distance"].default_value = 0.002
    wnt.links.new(wn.outputs["Fac"], wbmp.inputs["Height"])
    wnt.links.new(wbmp.outputs["Normal"], wbsdf.inputs["Normal"])
    mats["wall"] = wall

    # Acoustic wood wall slats (vertical walnut grain!)
    slat = _make_wood_mat("Lab_AcousticSlat", vertical=True)
    mats["wood_slat"] = slat

    # Window glass (thin architectural glass that transmits sunlight freely via Light Path trick!)
    gl, gnt, gbsdf = _new_mat("Lab_WindowGlass")
    # Mix Transparent BSDF for shadow/diffuse rays with Glass/Glossy for camera/reflection realism
    lp = gnt.nodes.new("ShaderNodeLightPath")
    transp = gnt.nodes.new("ShaderNodeBsdfTransparent")
    mixs = gnt.nodes.new("ShaderNodeMixShader")
    gbsdf.inputs["Base Color"].default_value = (0.92, 0.96, 0.98, 1.0)
    gbsdf.inputs["Roughness"].default_value = 0.02
    gbsdf.inputs["IOR"].default_value = 1.50
    gbsdf.inputs["Transmission Weight"].default_value = 0.95
    # Pass direct sun shadow rays straight through glass so sun casts crisp warm window mullions!
    gnt.links.new(lp.outputs["Is Shadow Ray"], mixs.inputs["Fac"])
    gnt.links.new(gbsdf.outputs["BSDF"], mixs.inputs[1])
    gnt.links.new(transp.outputs["BSDF"], mixs.inputs[2])
    out_node = [n for n in gnt.nodes if n.type == "OUTPUT_MATERIAL"][0]
    gnt.links.new(mixs.outputs["Shader"], out_node.inputs["Surface"])
    mats["window_glass"] = gl

    # Props: ceramic mug, paper notebook, terracotta pot, succulent foliage, warm practical LED
    mats["mug"] = _make_ceramic_mat("Prop_CoffeeMug", (0.14, 0.18, 0.21), 0.12, 0.22)
    mats["paper"] = _make_ceramic_mat("Prop_Paper", (0.76, 0.73, 0.66), 0.72, 0.85)
    mats["paper"].node_tree.nodes["Principled BSDF"].inputs["Coat Weight"].default_value = 0.0
    mats["terracotta"] = _make_ceramic_mat("Prop_Terracotta", (0.38, 0.16, 0.09), 0.68, 0.82)
    mats["terracotta"].node_tree.nodes["Principled BSDF"].inputs["Coat Weight"].default_value = 0.0
    mats["foliage"] = _make_ceramic_mat("Prop_Succulent", (0.10, 0.24, 0.11), 0.34, 0.50)
    mats["foliage"].node_tree.nodes["Principled BSDF"].inputs["Coat Weight"].default_value = 0.18
    mats["foliage"].node_tree.nodes["Principled BSDF"].inputs["Subsurface Weight"].default_value = 0.18
    mats["lamp_glow"] = _make_emission_mat("Lab_WarmLampGlow", (1.0, 0.74, 0.42), strength=14.0)
    mats["overhead_glow"] = _make_emission_mat("Lab_OverheadLED", (0.94, 0.96, 1.0), strength=8.5)
    mats["led_amber"] = _make_emission_mat("Lab_AmberLED", (1.0, 0.56, 0.12), strength=5.0)
    mats["led_green"] = _make_emission_mat("Lab_GreenLED", (0.18, 0.92, 0.42), strength=4.5)
    mats["chair_fabric"] = _make_silicone_mat("Lab_ChairMesh")

    return mats
