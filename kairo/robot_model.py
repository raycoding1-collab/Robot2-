"""Original KAIRO-04 humanoid laboratory dexterity robot geometry and rigid joint rig.

Design language:
  - Precision Swiss/Japanese robotics industrial design (original, no existing IP).
  - Warm satin white ceramic shells with genuine 1.5 mm panel parting seams over a dark
    anodized graphite structural chassis, brushed titanium rotary harmonic-drive actuators,
    matte black micro-textured silicone fingertip grip pads, recessed M3/M4 socket bolts,
    and cleanly routed matte-rubber flex cable conduits.
  - Sculpted anatomical head with compound-curved dark smoked optical visor backed by a
    100% opaque matte-black optical bulkhead (interior is NEVER visible) housing subtle
    dual-ring articulated optic iris lights.
  - Pure rigid-body parenting on Quaternion Empties with strictly positive +1 scale
    (left limbs use mirrored geometry with recalculated normals) for exact Cycles motion blur.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import geo
from .config import (
    CHEST_OFF,
    EYE_OFFSET,
    FINGERS,
    FINGER_ORDER,
    FORE_LEN,
    HEAD_OFF,
    NECK_OFF,
    PELVIS_POS,
    SHOULDER_OFF_R,
    SPINE_OFF,
    UPPER_LEN,
    WRIST_ROLL_AT,
)


def _mirror_bm_x(bm: bmesh.types.BMesh) -> bmesh.types.BMesh:
    """Mirror a bmesh across X=0 cleanly and fix face winding/normals (keeps +1 object scale)."""
    bmesh.ops.scale(bm, vec=(-1.0, 1.0, 1.0), verts=bm.verts)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _make_empty(name: str, parent=None, coll=None, loc=(0.0, 0.0, 0.0), size: float = 0.025):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = size
    e.empty_display_type = "PLAIN_AXES"
    e.rotation_mode = "QUATERNION"
    e.location = loc
    if parent is not None:
        e.parent = parent
    if coll is not None:
        geo.link_obj(e, coll)
    return e


# ==============================================================================
# Sculpted anatomical finger phalanx & hand builders
# ==============================================================================
def _build_phalanx_bm(length: float, width: float, thick: float, seg_idx: int, is_thumb: bool = False):
    """Build a sculpted anatomical robotic phalanx (ceramic dorsal shell + graphite core + silicone pad).

    Material slots:
      0 = ceramic white shell
      1 = dark anodized graphite structural frame / clevis
      2 = brushed titanium axle pin / trim
      3 = matte textured silicone grip pad
    """
    bm = bmesh.new()
    # Taper factors per segment (proximal -> middle -> distal)
    t_start = (1.00, 0.90, 0.80)[seg_idx]
    t_end = (0.91, 0.81, 0.64)[seg_idx]
    w0, w1 = 0.5 * width * t_start, 0.5 * width * t_end
    h0, h1 = 0.5 * thick * t_start, 0.5 * thick * t_end

    # 1. Recessed sleek graphite knuckle clevis barrel at y=0 (slim, flush with phalanx)
    kn_r = min(h0 * 0.80, 0.0054)
    kn_w = w0 * 1.82
    kn = geo.cylinder_bm(kn_r, kn_w, segs=20, bevel=0.0005, bevel_segs=2)
    geo.add_bm_into(bm, kn, mat_idx=1, xf=Matrix.Rotation(math.pi / 2, 4, "Y"))

    # Tiny brushed titanium axle pin end-caps protruding 0.35 mm on sides
    pin = geo.cylinder_bm(kn_r * 0.44, kn_w + 0.0007, segs=14, bevel=0.00025, bevel_segs=1)
    geo.add_bm_into(bm, pin, mat_idx=2, xf=Matrix.Rotation(math.pi / 2, 4, "Y"))

    # 2. Inner dark graphite structural bone spine spanning the phalanx
    y_a = kn_r * 0.55
    y_b = length - (kn_r * 0.50 if seg_idx < 2 else 0.004)
    spine_rings = [
        geo.superellipse_ring(w0 * 0.72, h0 * 0.66, y_a, power=2.4, n=16, z_off=-h0 * 0.08),
        geo.superellipse_ring(w1 * 0.70, h1 * 0.64, y_b, power=2.4, n=16, z_off=-h1 * 0.08),
    ]
    spine = geo.loft_rings_bm(spine_rings)
    geo.add_bm_into(bm, spine, mat_idx=1)

    # 3. Sculpted tapered white ceramic dorsal armor shell (clean sleek chamfers, NOT pill-boxes!)
    s_a = kn_r * 0.72
    s_b = length - (kn_r * 0.68 if seg_idx < 2 else 0.0012)
    n_prof = 8 if seg_idx == 2 else 6
    shell_rings = []
    for i in range(n_prof):
        u = i / (n_prof - 1)
        y = s_a + (s_b - s_a) * u
        # Gentle organic swell in the middle, crisp chamfer taper at ends
        end_taper = 1.0
        if i == 0:
            end_taper = 0.84
        elif i == n_prof - 1:
            end_taper = 0.36 if seg_idx == 2 else 0.84
        elif seg_idx == 2 and u > 0.72:
            # Smooth rounded anatomical fingertip dome
            v = (u - 0.72) / 0.28
            end_taper = math.sqrt(max(0.12, 1.0 - 0.86 * v * v))
        else:
            end_taper = 1.0 + 0.04 * math.sin(math.pi * u)
        rx = (w0 * (1.0 - u) + w1 * u) * end_taper
        rz = (h0 * (1.0 - u) + h1 * u) * end_taper
        # Slight dorsal crown (+Z) and slim palmar cut (-Z) so dark silicone pad sits flush underneath
        shell_rings.append(
            geo.superellipse_ring(
                rx,
                rz,
                y,
                power=2.8,
                n=22,
                z_off=rz * 0.14,
                flat_bottom=0.62 if seg_idx < 2 else 0.72,
            )
        )
    shell = geo.loft_rings_bm(shell_rings, cap_start=True, cap_end=True)
    geo.add_bm_into(bm, shell, mat_idx=0)

    # 4. Contoured matte-black silicone palmar grip pad (prominent on distal fingertip & middle phalanx)
    if seg_idx == 2:
        # Full contoured anatomical fingertip pad matching pad_point() at y ~ 0.68*L, z ~ -0.54*thick*0.84
        pad_a = s_a + 0.0020
        pad_b = length * 0.90
        pad_rings = []
        for i in range(7):
            u = i / 6.0
            y = pad_a + (pad_b - pad_a) * u
            env = math.sin(math.pi * (0.12 + 0.76 * u)) ** 0.55
            rx = (w0 * (1.0 - u) + w1 * u) * 0.80 * env
            rz = h1 * 0.50 * env
            z_c = -h1 * (0.48 + 0.14 * env)
            pad_rings.append(geo.superellipse_ring(rx, rz, y, power=2.2, n=18, z_off=z_c))
        pad = geo.loft_rings_bm(pad_rings)
        geo.add_bm_into(bm, pad, mat_idx=3)
        # Subtle micro-tread grip ribs across distal pad
        for r_u in (0.35, 0.52, 0.69):
            ry = pad_a + (pad_b - pad_a) * r_u
            rib = geo.rounded_box_bm(w1 * 1.15, 0.0007, 0.0007, 0.00025, 2)
            geo.add_bm_into(bm, rib, mat_idx=3, xf=Matrix.Translation((0.0, ry, -h1 * 0.94)))
    elif seg_idx == 1 or (seg_idx == 0 and not is_thumb):
        pad_len = (s_b - s_a) * 0.68
        pad_bm = geo.rounded_box_bm(w1 * 1.32, pad_len, h1 * 0.55, 0.0014, 3)
        geo.add_bm_into(bm, pad_bm, mat_idx=3, xf=Matrix.Translation((0.0, 0.5 * (s_a + s_b), -h1 * 0.62)))

    return bm


def _build_palm_bm(is_left: bool = False):
    """Build sculpted anatomical robotic metacarpal palm chassis (canonical right hand, mirrored if left)."""
    bm = bmesh.new()
    # 1. Dark anodized graphite inner structural chassis
    chassis_rings = [
        geo.superellipse_ring(0.025, 0.0115, 0.002, power=2.5, n=24, x_off=0.001),
        geo.superellipse_ring(0.032, 0.0125, 0.028, power=2.6, n=24, x_off=-0.001),
        geo.superellipse_ring(0.036, 0.0115, 0.062, power=2.8, n=24, x_off=0.001),
        geo.superellipse_ring(0.035, 0.0095, 0.083, power=2.8, n=24, x_off=0.002),
    ]
    geo.add_bm_into(bm, geo.loft_rings_bm(chassis_rings), mat_idx=1)

    # 2. Split sculpted ceramic dorsal armor shell (main metacarpal plate + subtle wrist cuff)
    dorsal_rings = [
        geo.superellipse_ring(0.0265, 0.0095, 0.010, power=3.0, n=26, z_off=0.0045, flat_bottom=0.35),
        geo.superellipse_ring(0.0345, 0.0112, 0.034, power=3.0, n=26, z_off=0.0052, flat_bottom=0.32),
        geo.superellipse_ring(0.0378, 0.0108, 0.066, power=3.2, n=26, z_off=0.0046, flat_bottom=0.30),
        geo.superellipse_ring(0.0355, 0.0082, 0.081, power=3.2, n=26, z_off=0.0038, flat_bottom=0.28),
    ]
    geo.add_bm_into(bm, geo.loft_rings_bm(dorsal_rings), mat_idx=0)

    # Recessed brushed-titanium dorsal telemetry accent bar & 4 micro M2.5 hex bolts
    bar = geo.rounded_box_bm(0.032, 0.016, 0.0022, 0.0006, 2)
    geo.add_bm_into(bm, bar, mat_idx=2, xf=Matrix.Translation((0.002, 0.044, 0.0156)))
    for bx, by in ((-0.018, 0.022), (0.020, 0.022), (-0.022, 0.070), (0.024, 0.070)):
        bolt = geo.hex_socket_bolt_bm(head_r=0.0017, head_h=0.0009)
        geo.add_bm_into(bm, bolt, mat_idx=2, xf=Matrix.Translation((bx, by, 0.0145)))

    # 3. Matte silicone thenar & hypothenar palmar grip cushions
    thenar = geo.rounded_box_bm(0.026, 0.038, 0.0075, 0.0032, 3)
    geo.add_bm_into(bm, thenar, mat_idx=3, xf=Matrix.Translation((-0.014, 0.038, -0.0105)))
    hypothenar = geo.rounded_box_bm(0.020, 0.044, 0.0068, 0.0030, 3)
    geo.add_bm_into(bm, hypothenar, mat_idx=3, xf=Matrix.Translation((0.016, 0.046, -0.0100)))

    # 4. Precision wrist gimbal sphere & titanium collar ring at y=0
    gimbal = bmesh.new()
    bmesh.ops.create_uvsphere(gimbal, u_segments=20, v_segments=14, radius=0.0175)
    geo.add_bm_into(bm, gimbal, mat_idx=1, xf=Matrix.Translation((0.0, -0.002, 0.0)))
    collar = geo.cylinder_bm(0.0215, 0.0065, segs=26, bevel=0.0009, bevel_segs=2)
    geo.add_bm_into(bm, collar, mat_idx=2, xf=Matrix.Translation((0.0, 0.006, 0.0)) @ Matrix.Rotation(math.pi / 2, 4, "X"))

    # Thumb CMC articulated socket boss
    tb = FINGERS["thumb"]["base"]
    boss = geo.cylinder_bm(0.0105, 0.014, segs=18, bevel=0.0015, bevel_segs=2)
    geo.add_bm_into(bm, boss, mat_idx=1, xf=Matrix.Translation(tb) @ Matrix.Rotation(0.6, 4, "Z"))

    if is_left:
        _mirror_bm_x(bm)
    return bm


# ==============================================================================
# Head, Visor, Torso & Limbs
# ==============================================================================
def _build_head_and_visor(head_empty, mats, coll):
    """Build KAIRO-04 sleek continuous ceramic helmet + sweeping smoked optical visor + opaque bulkhead + eye LEDs."""
    bm_skull = bmesh.new()
    # Sleek continuous aerodynamic cranium shell tapering cleanly around the sweeping front visor
    profile_y = [
        (-0.048, 0.068, 0.084, 0.114),
        (-0.022, 0.078, 0.096, 0.112),
        ( 0.018, 0.081, 0.100, 0.110),
        ( 0.062, 0.074, 0.090, 0.106),
        ( 0.098, 0.052, 0.064, 0.100),
        ( 0.112, 0.024, 0.030, 0.094),
    ]
    rings = [
        geo.superellipse_ring(rx, rz, y, power=2.5, n=36, z_off=zc, flat_bottom=0.88)
        for y, rx, rz, zc in profile_y
    ]
    geo.add_bm_into(bm_skull, geo.loft_rings_bm(rings), mat_idx=0)

    # Subtle sleek lower jaw / chin fairing neatly tucked under visor (no blocky protrusion)
    jaw_y = [
        (-0.076, 0.036, 0.015, 0.042),
        (-0.050, 0.054, 0.022, 0.046),
        (-0.018, 0.062, 0.026, 0.052),
        ( 0.018, 0.054, 0.026, 0.060),
    ]
    jaw_rings = [
        geo.superellipse_ring(rx, rz, y, power=2.4, n=30, z_off=zc)
        for y, rx, rz, zc in jaw_y
    ]
    geo.add_bm_into(bm_skull, geo.loft_rings_bm(jaw_rings), mat_idx=0)

    # Dark anodized graphite inner cranial chassis & recessed perimeter visor bezel
    inner_core = bmesh.new()
    bmesh.ops.create_uvsphere(inner_core, u_segments=28, v_segments=18, radius=0.069)
    bmesh.ops.scale(inner_core, vec=(0.94, 0.96, 1.12), verts=inner_core.verts)
    geo.add_bm_into(bm_skull, inner_core, mat_idx=1, xf=Matrix.Translation((0.0, 0.014, 0.102)))

    # Symmetrical CNC-machined temple acoustic/gimbal sensor pods (brushed titanium + graphite ring)
    for sx in (-1.0, 1.0):
        pod_base = geo.cylinder_bm(0.026, 0.016, segs=32, bevel=0.0022, bevel_segs=3)
        geo.add_bm_into(
            bm_skull,
            pod_base,
            mat_idx=1,
            xf=Matrix.Translation((sx * 0.078, 0.006, 0.102)) @ Matrix.Rotation(math.pi / 2, 4, "Y"),
        )
        pod_ring = geo.cylinder_bm(0.019, 0.019, segs=32, bevel=0.0016, bevel_segs=2)
        geo.add_bm_into(
            bm_skull,
            pod_ring,
            mat_idx=2,
            xf=Matrix.Translation((sx * 0.080, 0.006, 0.102)) @ Matrix.Rotation(math.pi / 2, 4, "Y"),
        )
        for ang in (0.0, math.pi / 2, math.pi, 1.5 * math.pi):
            bz = 0.102 + 0.0135 * math.sin(ang)
            by = 0.006 + 0.0135 * math.cos(ang)
            bolt = geo.hex_socket_bolt_bm(head_r=0.0017, head_h=0.0010)
            rot_y = Matrix.Rotation(sx * math.pi / 2, 4, "Y")
            geo.add_bm_into(bm_skull, bolt, mat_idx=2, xf=Matrix.Translation((sx * 0.0895, by, bz)) @ rot_y)

    neck_ring = geo.cylinder_bm(0.042, 0.022, segs=28, bevel=0.002, bevel_segs=2)
    geo.add_bm_into(bm_skull, neck_ring, mat_idx=1, xf=Matrix.Translation((0.0, 0.008, 0.018)))

    geo.new_mesh_obj(
        "KAIRO_HeadShell",
        bm_skull,
        mats=[mats["ceramic_head"], mats["graphite"], mats["titanium"]],
        parent=head_empty,
        coll=coll,
    )

    # 2. Sleek full-face compound-curved dark smoked optical visor backed by 100% opaque bulkhead
    def _visor_dome_bm(scale_r: float = 1.0, y_shift: float = 0.0):
        bm_v = bmesh.new()
        u_steps, v_steps = 40, 22
        verts_grid = []
        for iv in range(v_steps):
            phi = math.radians(-30.0 + 78.0 * iv / (v_steps - 1))
            row = []
            for iu in range(u_steps):
                theta = math.radians(-78.0 + 156.0 * iu / (u_steps - 1))
                rx = 0.0790 * scale_r
                ry = 0.0955 * scale_r
                rz = 0.0770 * scale_r
                x = rx * math.cos(phi) * math.sin(theta)
                y = -ry * math.cos(phi) * math.cos(theta) - 0.006 + y_shift
                z = 0.108 + rz * math.sin(phi)
                row.append(bm_v.verts.new(Vector((x, y, z))))
            verts_grid.append(row)
        for iv in range(v_steps - 1):
            for iu in range(u_steps - 1):
                bm_v.faces.new((
                    verts_grid[iv][iu],
                    verts_grid[iv][iu + 1],
                    verts_grid[iv + 1][iu + 1],
                    verts_grid[iv + 1][iu],
                ))
        bmesh.ops.recalc_face_normals(bm_v, faces=bm_v.faces)
        if scale_r >= 0.99:
            bmesh.ops.solidify(bm_v, geom=list(bm_v.faces), thickness=0.0020)
        return bm_v

    bulkhead_bm = _visor_dome_bm(scale_r=0.945, y_shift=0.0040)
    geo.new_mesh_obj("KAIRO_VisorBulkhead", bulkhead_bm, mats=[mats["optical_black"]], parent=head_empty, coll=coll)

    visor_bm = _visor_dome_bm(scale_r=1.0, y_shift=0.0)
    geo.new_mesh_obj("KAIRO_VisorGlass", visor_bm, mats=[mats["visor_glass"]], parent=head_empty, coll=coll)

    # 3. Subtle articulated dual-ring optical iris LEDs seated cleanly between bulkhead & smoked visor
    eye_objs = []
    for side, sx in (("R", -1.0), ("L", 1.0)):
        eye_e = _make_empty(
            f"KAIRO_Eye_{side}",
            parent=head_empty,
            coll=coll,
            loc=(sx * 0.0265, -0.0915, 0.112),
            size=0.012,
        )
        bm_eye = bmesh.new()
        n_seg = 30
        r_out, r_in = 0.0084, 0.0062
        ring_v_out = [bm_eye.verts.new((r_out * math.cos(2 * math.pi * k / n_seg), 0.0, r_out * math.sin(2 * math.pi * k / n_seg))) for k in range(n_seg)]
        ring_v_in = [bm_eye.verts.new((r_in * math.cos(2 * math.pi * k / n_seg), 0.0, r_in * math.sin(2 * math.pi * k / n_seg))) for k in range(n_seg)]
        for k in range(n_seg):
            kn = (k + 1) % n_seg
            f = bm_eye.faces.new((ring_v_out[k], ring_v_out[kn], ring_v_in[kn], ring_v_in[k]))
            f.material_index = 0
        pupil = geo.cylinder_bm(0.0025, 0.0008, segs=18)
        geo.add_bm_into(bm_eye, pupil, mat_idx=0, xf=Matrix.Rotation(math.pi / 2, 4, "X"))
        geo.new_mesh_obj(
            f"KAIRO_EyeOptic_{side}",
            bm_eye,
            mats=[mats["eye_led"]],
            parent=eye_e,
            coll=coll,
            rot=(0.0, 0.0, sx * 0.24),
        )
        eye_objs.append(eye_e)
    return eye_objs


def _build_torso_and_neck(pelvis_e, spine_e, chest_e, neck_e, mats, coll):
    """Build multi-panel KAIRO-04 seated torso, clavicle yoke, harmonic spine, and cable conduits."""
    # Pelvis / seated hip base
    bm_pelv = geo.loft_rings_bm([
        geo.superellipse_ring(0.155, 0.115, 0.0, power=2.6, n=32),
        geo.superellipse_ring(0.148, 0.108, 0.085, power=2.6, n=32),
    ])
    bmesh.ops.rotate(bm_pelv, matrix=Matrix.Rotation(math.pi / 2, 3, "X"), verts=bm_pelv.verts)
    geo.new_mesh_obj("KAIRO_Pelvis", bm_pelv, mats=[mats["graphite"]], parent=pelvis_e, coll=coll)

    # Articulated graphite lumbar/spine column
    bm_sp = bmesh.new()
    col = geo.cylinder_bm(0.068, 0.175, segs=28, bevel=0.006, bevel_segs=3)
    geo.add_bm_into(bm_sp, col, mat_idx=1, xf=Matrix.Translation((0.0, 0.008, 0.085)))
    for z in (0.045, 0.088, 0.128):
        rib = geo.rounded_box_bm(0.205, 0.152, 0.024, 0.008, 3)
        geo.add_bm_into(bm_sp, rib, mat_idx=1, xf=Matrix.Translation((0.0, 0.0, z)))
    ab_plate = geo.rounded_box_bm(0.168, 0.036, 0.115, 0.010, 4)
    geo.add_bm_into(bm_sp, ab_plate, mat_idx=0, xf=Matrix.Translation((0.0, -0.072, 0.082)))
    geo.new_mesh_obj("KAIRO_Spine", bm_sp, mats=[mats["ceramic_body"], mats["graphite"]], parent=spine_e, coll=coll)

    # Sculpted multi-panel anatomical upper torso (ergonomic wrap-around ceramic carapace)
    bm_ch = bmesh.new()
    core_rings = [
        geo.superellipse_ring(0.118, 0.084, 0.00, power=2.5, n=32),
        geo.superellipse_ring(0.156, 0.098, 0.13, power=2.7, n=32),
        geo.superellipse_ring(0.178, 0.096, 0.25, power=2.9, n=32),
        geo.superellipse_ring(0.142, 0.074, 0.312, power=2.5, n=32),
    ]
    core_bm = geo.loft_rings_bm(core_rings)
    bmesh.ops.rotate(core_bm, matrix=Matrix.Rotation(math.pi / 2, 3, "X"), verts=core_bm.verts)
    geo.add_bm_into(bm_ch, core_bm, mat_idx=1)

    for sx in (-1.0, 1.0):
        pec_rings = []
        z_levels = [
            (0.055, 0.052, 0.044, -0.052, 0.064),
            (0.115, 0.072, 0.062, -0.060, 0.082),
            (0.195, 0.082, 0.068, -0.064, 0.090),
            (0.268, 0.076, 0.058, -0.054, 0.088),
            (0.302, 0.054, 0.040, -0.032, 0.084),
        ]
        for z_h, rx, ry, y_off, x_ctr in z_levels:
            ring = geo.superellipse_ring(
                rx,
                ry,
                z_h,
                power=3.0,
                n=28,
                z_off=y_off,
                x_off=sx * x_ctr,
                flat_bottom=0.85,
            )
            pec_rings.append([Vector((p.x, p.z, p.y)) for p in ring])
        pec_bm = geo.loft_rings_bm(pec_rings, cap_start=True, cap_end=True)
        bmesh.ops.recalc_face_normals(pec_bm, faces=pec_bm.faces)
        geo.add_bm_into(bm_ch, pec_bm, mat_idx=0)

        # Low-profile sleek horizontal clavicle/trapezius shoulder fairing (NO tall vertical pillows!)
        trap = geo.rounded_box_bm(0.115, 0.110, 0.028, 0.011, 3)
        geo.add_bm_into(
            bm_ch,
            trap,
            mat_idx=0,
            xf=Matrix.Translation((sx * 0.118, 0.008, 0.308)) @ Matrix.Rotation(-sx * 0.16, 4, "Y"),
        )

        # Exposed brushed-titanium recessed fasteners along pectoral bevel & clavicle
        for fx, fy, fz in ((0.032, -0.114, 0.274), (0.128, -0.104, 0.270), (0.036, -0.116, 0.120)):
            bolt = geo.hex_socket_bolt_bm(head_r=0.0028, head_h=0.0014)
            geo.add_bm_into(
                bm_ch,
                bolt,
                mat_idx=2,
                xf=Matrix.Translation((sx * fx, fy, fz)) @ Matrix.Rotation(math.pi / 2, 4, "X"),
            )

    # Rear sculpted ergonomic ceramic dorsal backpack + cooling vent louvers
    back_rings = [
        geo.superellipse_ring(0.118, 0.042, 0.065, power=2.8, n=28, z_off=0.072),
        geo.superellipse_ring(0.146, 0.052, 0.175, power=3.0, n=28, z_off=0.078),
        geo.superellipse_ring(0.132, 0.046, 0.292, power=2.8, n=28, z_off=0.068),
    ]
    back_bm = geo.loft_rings_bm([[Vector((p.x, p.z, p.y)) for p in r] for r in back_rings])
    bmesh.ops.recalc_face_normals(back_bm, faces=back_bm.faces)
    geo.add_bm_into(bm_ch, back_bm, mat_idx=0)
    for vz in (0.14, 0.17, 0.20, 0.23):
        louver = geo.rounded_box_bm(0.14, 0.014, 0.008, 0.002, 1)
        geo.add_bm_into(bm_ch, louver, mat_idx=1, xf=Matrix.Translation((0.0, 0.124, vz)))

    # Sculpted dark-graphite collar yoke & machined titanium clavicle linkages
    yoke = geo.cylinder_bm(0.066, 0.028, segs=32, bevel=0.004, bevel_segs=2)
    geo.add_bm_into(bm_ch, yoke, mat_idx=1, xf=Matrix.Translation((0.0, 0.004, 0.318)))
    for sx in (-1.0, 1.0):
        p0 = Vector((sx * 0.056, -0.042, 0.312))
        p1 = Vector((sx * 0.176, -0.008, 0.294))
        strut = geo.tube_along_points_bm([p0, p0.lerp(p1, 0.5) + Vector((0, -0.010, 0.008)), p1], radius=0.010, segs=16)
        geo.add_bm_into(bm_ch, strut, mat_idx=2)
        grom = geo.cylinder_bm(0.012, 0.016, segs=18, bevel=0.002, bevel_segs=2)
        geo.add_bm_into(bm_ch, grom, mat_idx=1, xf=Matrix.Translation((sx * 0.042, 0.012, 0.326)))

    # Recessed dark-graphite sternum bridge & warm-amber telemetry LED strip
    stern = geo.rounded_box_bm(0.024, 0.016, 0.175, 0.004, 2)
    geo.add_bm_into(bm_ch, stern, mat_idx=1, xf=Matrix.Translation((0.0, -0.104, 0.195)))
    led = geo.rounded_box_bm(0.0045, 0.004, 0.032, 0.001, 2)
    geo.add_bm_into(bm_ch, led, mat_idx=3, xf=Matrix.Translation((0.0, -0.112, 0.236)))

    # Titanium serial badge backing frame flush on upper-left chest (+X side)
    badge = geo.rounded_box_bm(0.072, 0.006, 0.026, 0.0015, 2)
    geo.add_bm_into(bm_ch, badge, mat_idx=2, xf=Matrix.Translation((0.088, -0.123, 0.238)) @ Matrix.Rotation(0.10, 4, "Z"))

    # Shoulder harmonic-drive mount housings + precision concentric gear ring + socket bolts
    for sx in (-1.0, 1.0):
        sh_pos = Vector((sx * abs(SHOULDER_OFF_R.x), SHOULDER_OFF_R.y, SHOULDER_OFF_R.z))
        drum = geo.cylinder_bm(0.052, 0.064, segs=32, bevel=0.004, bevel_segs=3)
        geo.add_bm_into(
            bm_ch,
            drum,
            mat_idx=1,
            xf=Matrix.Translation(sh_pos) @ Matrix.Rotation(math.pi / 2, 4, "Y"),
        )
        cap = geo.cylinder_bm(0.038, 0.070, segs=28, bevel=0.0025, bevel_segs=2)
        geo.add_bm_into(
            bm_ch,
            cap,
            mat_idx=2,
            xf=Matrix.Translation(sh_pos) @ Matrix.Rotation(math.pi / 2, 4, "Y"),
        )
        for bi in range(6):
            ba = 2.0 * math.pi * bi / 6.0
            bp = sh_pos + Vector((sx * 0.0355, 0.027 * math.cos(ba), 0.027 * math.sin(ba)))
            b = geo.hex_socket_bolt_bm(head_r=0.0024, head_h=0.0012)
            geo.add_bm_into(bm_ch, b, mat_idx=2, xf=Matrix.Translation(bp) @ Matrix.Rotation(sx * math.pi / 2, 4, "Y"))

    geo.new_mesh_obj(
        "KAIRO_Chest",
        bm_ch,
        mats=[mats["ceramic_body"], mats["graphite"], mats["titanium"], mats["status_led"]],
        parent=chest_e,
        coll=coll,
    )

    # Crisp UV-mapped laser-etched "KAIRO-04" titanium serial plate (viewer looking toward +Y: -X is LEFT, +X is RIGHT!)
    bm_bdg = bmesh.new()
    uv_l = bm_bdg.loops.layers.uv.new("UVMap")
    bw, bh = 0.066, 0.022
    v0 = bm_bdg.verts.new((-0.5 * bw, 0.0, -0.5 * bh))
    v1 = bm_bdg.verts.new((+0.5 * bw, 0.0, -0.5 * bh))
    v2 = bm_bdg.verts.new((+0.5 * bw, 0.0, +0.5 * bh))
    v3 = bm_bdg.verts.new((-0.5 * bw, 0.0, +0.5 * bh))
    f_b = bm_bdg.faces.new((v0, v1, v2, v3))
    # Looking from -Y toward +Y: -X is viewer's LEFT (u=0), +X is viewer's RIGHT (u=1)!
    for loop, uv in zip(f_b.loops, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
        loop[uv_l].uv = uv
    geo.new_mesh_obj(
        "KAIRO_SerialBadgePlate",
        bm_bdg,
        mats=[mats["badge"]],
        parent=chest_e,
        coll=coll,
        smooth=False,
        loc=(0.088, -0.1266, 0.238),
        rot=(0.0, 0.0, 0.10),
    )

    # Articulated cervical neck column + dual matte-rubber braided flex cables anchored cleanly at BOTH ends!
    bm_nk = bmesh.new()
    cyl = geo.cylinder_bm(0.036, 0.095, segs=28, bevel=0.003, bevel_segs=2)
    geo.add_bm_into(bm_nk, cyl, mat_idx=1, xf=Matrix.Translation((0.0, 0.0, 0.045)))
    collar = geo.cylinder_bm(0.049, 0.016, segs=32, bevel=0.0025, bevel_segs=2)
    geo.add_bm_into(bm_nk, collar, mat_idx=2, xf=Matrix.Translation((0.0, 0.0, 0.008)))
    for sx in (-1.0, 1.0):
        # Anchored bottom inside chest grommet (z=-0.008) & top inside head collar (z=+0.090) — ZERO floating ends!
        pts = [
            Vector((sx * 0.042, 0.012, -0.008)),
            Vector((sx * 0.046, 0.015, 0.022)),
            Vector((sx * 0.046, 0.014, 0.056)),
            Vector((sx * 0.038, 0.010, 0.090)),
        ]
        geo.add_bm_into(bm_nk, geo.tube_along_points_bm(pts, radius=0.0048, segs=12), mat_idx=3)
    geo.new_mesh_obj(
        "KAIRO_Neck",
        bm_nk,
        mats=[mats["ceramic_body"], mats["graphite"], mats["titanium"], mats["cable"]],
        parent=neck_e,
        coll=coll,
    )


def _build_arm_meshes(side: str, upper_e, fore_e, wroll_e, hand_e, finger_empties, mats, coll):
    """Build sculpted ceramic + graphite + titanium arm, wrist gimbal, cables, and dexterous hand."""
    is_left = side == "L"
    sx = 1.0 if is_left else -1.0

    # 1. Upper arm (along +Y from 0 to UPPER_LEN)
    bm_up = bmesh.new()
    # Deltoid spherical joint cap + bicep dual clamshell ceramic housing
    sph = bmesh.new()
    bmesh.ops.create_uvsphere(sph, u_segments=24, v_segments=16, radius=0.048)
    geo.add_bm_into(bm_up, sph, mat_idx=1)

    up_core = geo.cylinder_bm(0.030, UPPER_LEN * 0.86, segs=22, bevel=0.003, bevel_segs=2)
    geo.add_bm_into(
        bm_up,
        up_core,
        mat_idx=1,
        xf=Matrix.Translation((0.0, UPPER_LEN * 0.50, 0.0)) @ Matrix.Rotation(math.pi / 2, 4, "X"),
    )
    # Sculpted bicep ceramic shell
    bi_rings = [
        geo.superellipse_ring(0.046, 0.044, 0.025, power=2.6, n=28),
        geo.superellipse_ring(0.049, 0.046, 0.110, power=2.8, n=28),
        geo.superellipse_ring(0.043, 0.040, 0.215, power=2.8, n=28),
        geo.superellipse_ring(0.036, 0.035, UPPER_LEN - 0.034, power=2.5, n=28),
    ]
    geo.add_bm_into(bm_up, geo.loft_rings_bm(bi_rings), mat_idx=0)
    # Recessed dark titanium trim collar near elbow
    ring = geo.cylinder_bm(0.0385, 0.012, segs=28, bevel=0.0015, bevel_segs=2)
    geo.add_bm_into(
        bm_up,
        ring,
        mat_idx=2,
        xf=Matrix.Translation((0.0, UPPER_LEN - 0.044, 0.0)) @ Matrix.Rotation(math.pi / 2, 4, "X"),
    )
    # Routed matte-rubber flex cable running neatly along medial upper arm
    c_pts = [
        Vector((0.0, 0.045, -0.046)),
        Vector((0.0, 0.140, -0.050)),
        Vector((0.0, UPPER_LEN - 0.035, -0.042)),
    ]
    geo.add_bm_into(bm_up, geo.tube_along_points_bm(c_pts, radius=0.0045, segs=10), mat_idx=3)
    geo.new_mesh_obj(
        f"KAIRO_UpperArm_{side}",
        bm_up,
        mats=[mats["ceramic_arm"], mats["graphite"], mats["titanium"], mats["cable"]],
        parent=upper_e,
        coll=coll,
    )

    # 2. Forearm proximal section (elbow hinge at y=0 to WRIST_ROLL_AT)
    bm_fo = bmesh.new()
    # Exposed CNC harmonic-drive elbow actuator cylinder (axis along +X = hinge axis)
    el_drum = geo.cylinder_bm(0.035, 0.068, segs=30, bevel=0.003, bevel_segs=2)
    geo.add_bm_into(bm_fo, el_drum, mat_idx=1, xf=Matrix.Rotation(math.pi / 2, 4, "Y"))
    el_cap = geo.cylinder_bm(0.024, 0.074, segs=26, bevel=0.002, bevel_segs=2)
    geo.add_bm_into(bm_fo, el_cap, mat_idx=2, xf=Matrix.Rotation(math.pi / 2, 4, "Y"))
    for ex in (-0.0375, 0.0375):
        bolt = geo.hex_socket_bolt_bm(head_r=0.0030, head_h=0.0014)
        geo.add_bm_into(
            bm_fo,
            bolt,
            mat_idx=2,
            xf=Matrix.Translation((ex, 0.0, 0.0)) @ Matrix.Rotation(math.copysign(math.pi / 2, ex), 4, "Y"),
        )

    # Sculpted forearm proximal ceramic shell
    fo_rings = [
        geo.superellipse_ring(0.035, 0.037, 0.026, power=2.6, n=28),
        geo.superellipse_ring(0.039, 0.041, 0.075, power=2.8, n=28),
        geo.superellipse_ring(0.035, 0.036, WRIST_ROLL_AT - 0.008, power=2.8, n=28),
    ]
    geo.add_bm_into(bm_fo, geo.loft_rings_bm(fo_rings), mat_idx=0)
    # Dark graphite separation groove at wrist pronation joint
    sep = geo.cylinder_bm(0.031, 0.018, segs=28, bevel=0.0015, bevel_segs=2)
    geo.add_bm_into(
        bm_fo,
        sep,
        mat_idx=1,
        xf=Matrix.Translation((0.0, WRIST_ROLL_AT, 0.0)) @ Matrix.Rotation(math.pi / 2, 4, "X"),
    )
    geo.new_mesh_obj(
        f"KAIRO_Forearm_{side}",
        bm_fo,
        mats=[mats["ceramic_arm"], mats["graphite"], mats["titanium"]],
        parent=fore_e,
        coll=coll,
    )

    # 3. Distal forearm / wrist-roll cuff (from WRIST_ROLL_AT to FORE_LEN)
    cuff_len = FORE_LEN - WRIST_ROLL_AT
    bm_wr = bmesh.new()
    wr_rings = [
        geo.superellipse_ring(0.034, 0.035, 0.006, power=2.8, n=28),
        geo.superellipse_ring(0.031, 0.030, cuff_len * 0.52, power=2.8, n=28),
        geo.superellipse_ring(0.026, 0.021, cuff_len - 0.010, power=2.8, n=28),
    ]
    geo.add_bm_into(bm_wr, geo.loft_rings_bm(wr_rings), mat_idx=0)
    # Titanium wrist cuff bezel & micro status ring
    bezel = geo.cylinder_bm(0.0245, 0.010, segs=28, bevel=0.0012, bevel_segs=2)
    geo.add_bm_into(
        bm_wr,
        bezel,
        mat_idx=2,
        xf=Matrix.Translation((0.0, cuff_len - 0.007, 0.0)) @ Matrix.Rotation(math.pi / 2, 4, "X"),
    )
    # Micro fasteners on wrist cuff
    for ang in (0.6, math.pi - 0.6, math.pi + 0.6, -0.6):
        bolt = geo.hex_socket_bolt_bm(head_r=0.0018, head_h=0.0009)
        bx, bz = 0.030 * math.cos(ang), 0.029 * math.sin(ang)
        rot_q = Vector((0, 0, 1)).rotation_difference(Vector((bx, 0, bz)).normalized()).to_matrix().to_4x4()
        geo.add_bm_into(bm_wr, bolt, mat_idx=2, xf=Matrix.Translation((bx, cuff_len * 0.45, bz)) @ rot_q)
    geo.new_mesh_obj(
        f"KAIRO_WristCuff_{side}",
        bm_wr,
        mats=[mats["ceramic_hand"], mats["graphite"], mats["titanium"]],
        parent=wroll_e,
        coll=coll,
    )

    # 4. Palm mesh
    hand_mats = [mats["ceramic_hand"], mats["graphite"], mats["titanium"], mats["silicone"]]
    palm_bm = _build_palm_bm(is_left=is_left)
    geo.new_mesh_obj(f"KAIRO_Palm_{side}", palm_bm, mats=hand_mats, parent=hand_e, coll=coll)

    # 5. Sculpted fingers (3 phalanges per finger)
    for fn in FINGER_ORDER:
        fcfg = FINGERS[fn]
        mcp_e, pip_e, dip_e = finger_empties[fn]
        for seg_idx, (seg_e, L) in enumerate(zip((mcp_e, pip_e, dip_e), fcfg["lens"])):
            p_bm = _build_phalanx_bm(L, fcfg["width"], fcfg["thick"], seg_idx, is_thumb=(fn == "thumb"))
            if is_left:
                _mirror_bm_x(p_bm)
            geo.new_mesh_obj(
                f"KAIRO_{side}_{fn}_s{seg_idx}",
                p_bm,
                mats=hand_mats,
                parent=seg_e,
                coll=coll,
            )


def build_robot(coll, mats: Dict[str, object]) -> Dict[str, object]:
    """Build the complete KAIRO-04 robot hierarchy and return a dict of all animated joint Empties."""
    root = _make_empty("KAIRO_Root", coll=coll, size=0.08)
    pelvis = _make_empty("KAIRO_Pelvis_J", parent=root, coll=coll, loc=PELVIS_POS, size=0.05)
    spine = _make_empty("KAIRO_Spine_J", parent=pelvis, coll=coll, loc=SPINE_OFF, size=0.04)
    chest = _make_empty("KAIRO_Chest_J", parent=spine, coll=coll, loc=CHEST_OFF, size=0.05)
    neck = _make_empty("KAIRO_Neck_J", parent=chest, coll=coll, loc=NECK_OFF, size=0.03)
    head = _make_empty("KAIRO_Head_J", parent=neck, coll=coll, loc=HEAD_OFF, size=0.04)

    _build_torso_and_neck(pelvis, spine, chest, neck, mats, coll)
    eye_objs = _build_head_and_visor(head, mats, coll)

    rig: Dict[str, object] = {
        "root": root,
        "pelvis": pelvis,
        "spine": spine,
        "chest": chest,
        "neck": neck,
        "head": head,
        "eyes": eye_objs,
        "arms": {},
    }

    for side in ("R", "L"):
        is_left = side == "L"
        # World-driven arm joint empties (directly keyframed from analytic IK for zero drift)
        upper = _make_empty(f"KAIRO_Upper_{side}_J", parent=root, coll=coll, size=0.035)
        fore = _make_empty(f"KAIRO_Fore_{side}_J", parent=root, coll=coll, size=0.032)
        wroll = _make_empty(f"KAIRO_WRoll_{side}_J", parent=root, coll=coll, size=0.028)
        hand = _make_empty(f"KAIRO_Hand_{side}_J", parent=root, coll=coll, size=0.030)

        fingers_d = {}
        for fn in FINGER_ORDER:
            fcfg = FINGERS[fn]
            bx, by, bz = fcfg["base"]
            base_loc = (-bx if is_left else bx, by, bz)
            mcp = _make_empty(f"KAIRO_{side}_{fn}_mcp", parent=hand, coll=coll, loc=base_loc, size=0.008)
            pip = _make_empty(f"KAIRO_{side}_{fn}_pip", parent=mcp, coll=coll, loc=(0.0, fcfg["lens"][0], 0.0), size=0.007)
            dip = _make_empty(f"KAIRO_{side}_{fn}_dip", parent=pip, coll=coll, loc=(0.0, fcfg["lens"][1], 0.0), size=0.006)
            fingers_d[fn] = (mcp, pip, dip)

        _build_arm_meshes(side, upper, fore, wroll, hand, fingers_d, mats, coll)
        rig["arms"][side] = {
            "upper": upper,
            "fore": fore,
            "wroll": wroll,
            "hand": hand,
            "fingers": fingers_d,
        }

    return rig
