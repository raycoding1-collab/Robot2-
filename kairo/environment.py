"""Sunlit robotics laboratory environment, lived-in workbench props, architectural windows & lighting.

Designed specifically for rich foreground/midground/background cinematic depth:
  - Physical sky + golden late-afternoon sun streaming UNOBSTRUCTED through tall mullioned
    industrial windows on the robot's left/rear-left (+X / +Y corner).
  - Layered exterior view (courtyard parapet + sunlit trees placed strictly below/beside
    the sun ray path so exterior geometry NEVER blocks direct sunlight!).
  - Acoustic walnut wood-slat accent wall, rear equipment rack with glowing oscilloscope,
    benchtop power supply, parts bins, and warm recessed shelf accent strip.
  - Oiled walnut workbench + metric ESD mat + lived-in lab props cleanly outside hand sweep.
  - Practical warm 3100 K desk lamp + overhead linear LED + invisible glossy-free soft bounce card.
"""
from __future__ import annotations

import math
from typing import Dict

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

from . import geo
from .config import MAT_THICK, SURFACE_Z, TABLE_TOP_Z


def _uv_plane_obj(name: str, sx: float, sy: float, mat, coll, loc=(0, 0, 0), rot=(0, 0, 0)):
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    v0 = bm.verts.new((-0.5 * sx, -0.5 * sy, 0.0))
    v1 = bm.verts.new((+0.5 * sx, -0.5 * sy, 0.0))
    v2 = bm.verts.new((+0.5 * sx, +0.5 * sy, 0.0))
    v3 = bm.verts.new((-0.5 * sx, +0.5 * sy, 0.0))
    f = bm.faces.new((v0, v1, v2, v3))
    uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for loop, uv in zip(f.loops, uvs):
        loop[uv_layer].uv = uv
    return geo.new_mesh_obj(name, bm, mats=[mat], coll=coll, smooth=False, loc=loc, rot=rot)


def _setup_physical_world():
    """Physical multiple-scattering sky & late-afternoon sun shining cleanly through left-rear windows."""
    sc = bpy.context.scene
    w = bpy.data.worlds.new("Lab_PhysicalSky")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = True
    sky.sun_size = math.radians(1.0)
    sky.sun_intensity = 2.45
    # Golden late-afternoon sun streaming cleanly through the tall left (+X) mullioned windows!
    sky.sun_elevation = math.radians(18.5)
    sky.sun_rotation = math.radians(-102.0)
    sky.air_density = 1.45
    sky.ozone_density = 1.5

    bg.inputs["Strength"].default_value = 0.78
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    sc.world = w
    return sky


def _build_workbench_and_props(coll, mats: Dict[str, object]):
    """Build walnut workbench, metric ESD mat, and rich lived-in robotics lab desk props."""
    # 1. Solid oiled walnut tabletop (1.84 m wide x 0.88 m deep x 0.042 m thick)
    desk_cy = -0.58
    top_bm = geo.rounded_box_bm(1.86, 0.88, 0.042, 0.0045, 4)
    geo.new_mesh_obj("Desk_WalnutTop", top_bm, mats=[mats["walnut"]], coll=coll, loc=(0.0, desk_cy, TABLE_TOP_Z - 0.021))

    # Brushed aluminum front edge extrusion rail & heavy dark-steel legs
    rail_bm = geo.rounded_box_bm(1.862, 0.012, 0.026, 0.0018, 2)
    geo.new_mesh_obj("Desk_FrontRail", rail_bm, mats=[mats["aluminum"]], coll=coll, loc=(0.0, desk_cy + 0.440, TABLE_TOP_Z - 0.022))
    for sx in (-0.82, 0.82):
        for sy in (-0.22, -0.94):
            leg = geo.rounded_box_bm(0.052, 0.052, TABLE_TOP_Z - 0.042, 0.004, 2)
            geo.new_mesh_obj(f"Desk_Leg_{sx:+.1f}_{sy:+.1f}", leg, mats=[mats["dark_steel"]], coll=coll, loc=(sx, sy, 0.5 * (TABLE_TOP_Z - 0.042)))

    # 2. ESD anti-static metric cutting mat (0.76 m x 0.42 m x 2.5 mm) with crisp UV grid
    mat_cy = -0.43
    mat_bm = geo.rounded_box_bm(0.76, 0.42, MAT_THICK, 0.0008, 2)
    uv_layer = mat_bm.loops.layers.uv.new("UVMap")
    for f in mat_bm.faces:
        for loop in f.loops:
            co = loop.vert.co
            u = (co.x / 0.76) + 0.5
            v = (co.y / 0.42) + 0.5
            loop[uv_layer].uv = (u, v)
    geo.new_mesh_obj("Desk_ESDMat", mat_bm, mats=[mats["esd_mat"]], coll=coll, loc=(0.0, mat_cy, TABLE_TOP_Z + 0.5 * MAT_THICK))

    # 3. Articulated anodized aluminum warm bench lamp placed cleanly on far-right rear corner
    #    (x = -0.76, y = -0.24) — exact endpoint joints so rods never poke through the shade!
    lx, ly = -0.76, -0.24
    bm_lamp = bmesh.new()
    base = geo.cylinder_bm(0.082, 0.018, segs=32, bevel=0.003, bevel_segs=2)
    geo.add_bm_into(bm_lamp, base, mat_idx=0, xf=Matrix.Translation((0, 0, 0.009)))
    p_b = Vector((0.0, 0.0, 0.018))
    p_e = Vector((-0.05, 0.02, 0.34))
    p_s = Vector((0.10, -0.06, 0.49))
    geo.add_bm_into(bm_lamp, geo.tube_along_points_bm([p_b, p_e], radius=0.009, segs=16), mat_idx=1)
    elbow_j = bmesh.new()
    bmesh.ops.create_uvsphere(elbow_j, u_segments=16, v_segments=12, radius=0.014)
    geo.add_bm_into(bm_lamp, elbow_j, mat_idx=0, xf=Matrix.Translation(p_e))
    geo.add_bm_into(bm_lamp, geo.tube_along_points_bm([p_e, p_s], radius=0.008, segs=16), mat_idx=1)
    # Lamp shade socket attaches cleanly at p_s and aims down toward workbench center
    aim_dir = (Vector((0.68, -0.14, -0.45))).normalized()
    q_sh = Vector((0.0, 0.0, -1.0)).rotation_difference(aim_dir).to_matrix().to_4x4()
    shade_ctr = p_s + aim_dir * 0.052
    shade = bmesh.new()
    bmesh.ops.create_cone(shade, cap_ends=True, cap_tris=False, segments=32, radius1=0.068, radius2=0.022, depth=0.105)
    geo.add_bm_into(bm_lamp, shade, mat_idx=0, xf=Matrix.Translation(shade_ctr) @ q_sh)
    bulb = geo.cylinder_bm(0.048, 0.008, segs=24)
    geo.add_bm_into(bm_lamp, bulb, mat_idx=2, xf=Matrix.Translation(p_s + aim_dir * 0.088) @ q_sh)
    geo.new_mesh_obj("Prop_BenchLamp", bm_lamp, mats=[mats["dark_steel"], mats["aluminum"], mats["lamp_glow"]], coll=coll, loc=(lx, ly, TABLE_TOP_Z))

    ld = bpy.data.lights.new("Light_BenchLamp", type="SPOT")
    ld.energy = 32.0
    ld.color = (1.0, 0.72, 0.40)
    ld.spot_size = math.radians(76.0)
    ld.spot_blend = 0.55
    ld.shadow_soft_size = 0.045
    lo = bpy.data.objects.new("Light_BenchLamp", ld)
    lo.location = Vector((lx, ly, TABLE_TOP_Z)) + p_s + aim_dir * 0.11
    aim = (Vector((-0.06, -0.36, SURFACE_Z)) - lo.location).to_track_quat("-Z", "Y")
    lo.rotation_euler = aim.to_euler()
    geo.link_obj(lo, coll)

    # 4. Precision stainless vernier caliper resting cleanly on left side of mat (x=+0.34, y=-0.46)
    bm_cal = bmesh.new()
    bar = geo.rounded_box_bm(0.185, 0.016, 0.0035, 0.0006, 2)
    geo.add_bm_into(bm_cal, bar, mat_idx=0, xf=Matrix.Translation((0, 0, 0.0018)))
    jaw1 = geo.rounded_box_bm(0.014, 0.048, 0.0038, 0.0006, 2)
    geo.add_bm_into(bm_cal, jaw1, mat_idx=0, xf=Matrix.Translation((-0.086, -0.016, 0.0020)))
    jaw2 = geo.rounded_box_bm(0.014, 0.048, 0.0038, 0.0006, 2)
    geo.add_bm_into(bm_cal, jaw2, mat_idx=0, xf=Matrix.Translation((-0.029, -0.016, 0.0020)))
    lcd = geo.rounded_box_bm(0.058, 0.026, 0.0075, 0.0012, 2)
    geo.add_bm_into(bm_cal, lcd, mat_idx=1, xf=Matrix.Translation((-0.012, 0.002, 0.0042)))
    geo.new_mesh_obj("Prop_Caliper", bm_cal, mats=[mats["aluminum"], mats["dark_steel"]], coll=coll, loc=(0.35, -0.49, SURFACE_Z), rot=(0, 0, -0.36))

    # 5. Matte slate ceramic coffee mug (left-front: x=+0.50, y=-0.38)
    bm_mug = bmesh.new()
    outer = geo.cylinder_bm(0.041, 0.094, segs=32, bevel=0.0035, bevel_segs=3)
    geo.add_bm_into(bm_mug, outer, mat_idx=0, xf=Matrix.Translation((0, 0, 0.047)))
    # Handle arc
    h_pts = [
        Vector((0.038, 0, 0.072)),
        Vector((0.062, 0, 0.068)),
        Vector((0.066, 0, 0.048)),
        Vector((0.060, 0, 0.028)),
        Vector((0.038, 0, 0.024)),
    ]
    geo.add_bm_into(bm_mug, geo.tube_along_points_bm(h_pts, radius=0.0062, segs=12), mat_idx=0)
    # Dark espresso liquid surface inset at top
    liq = geo.cylinder_bm(0.0355, 0.003, segs=28)
    geo.add_bm_into(bm_mug, liq, mat_idx=1, xf=Matrix.Translation((0, 0, 0.088)))
    geo.new_mesh_obj("Prop_CoffeeMug", bm_mug, mats=[mats["mug"], mats["optical_black"]], coll=coll, loc=(0.52, -0.36, TABLE_TOP_Z), rot=(0, 0, 0.72))

    # 6. Hardcover lab notebook + machined brass/aluminum pen (right side: x=-0.47, y=-0.46)
    bm_nb = bmesh.new()
    cov = geo.rounded_box_bm(0.165, 0.225, 0.015, 0.0025, 2)
    geo.add_bm_into(bm_nb, cov, mat_idx=0, xf=Matrix.Translation((0, 0, 0.0075)))
    pgs = geo.rounded_box_bm(0.158, 0.218, 0.012, 0.0008, 1)
    geo.add_bm_into(bm_nb, pgs, mat_idx=1, xf=Matrix.Translation((0.003, 0, 0.0095)))
    pen = geo.cylinder_bm(0.0046, 0.142, segs=16, bevel=0.001, bevel_segs=1)
    geo.add_bm_into(bm_nb, pen, mat_idx=2, xf=Matrix.Translation((0.022, 0.015, 0.0195)) @ Matrix.Rotation(math.pi / 2, 4, "X") @ Matrix.Rotation(0.24, 4, "Z"))
    geo.new_mesh_obj("Prop_LabNotebook", bm_nb, mats=[mats["dark_steel"], mats["paper"], mats["brass"]], coll=coll, loc=(-0.48, -0.46, TABLE_TOP_Z), rot=(0, 0, 0.22))

    # 7. Anodized magnetic parts tray with tiny M3 hex bolts (right-rear: x=-0.34, y=-0.62)
    bm_tray = bmesh.new()
    dish = geo.cylinder_bm(0.052, 0.014, segs=32, bevel=0.003, bevel_segs=3)
    geo.add_bm_into(bm_tray, dish, mat_idx=0, xf=Matrix.Translation((0, 0, 0.007)))
    for bx, by in ((-0.014, 0.008), (0.010, -0.012), (0.018, 0.012), (-0.005, -0.018), (0.002, 0.005)):
        bolt = geo.hex_socket_bolt_bm(head_r=0.0034, head_h=0.0045)
        geo.add_bm_into(bm_tray, bolt, mat_idx=1, xf=Matrix.Translation((bx, by, 0.014)) @ Matrix.Rotation(0.8, 4, "X"))
    geo.new_mesh_obj("Prop_PartsTray", bm_tray, mats=[mats["dark_steel"], mats["titanium"]], coll=coll, loc=(-0.34, -0.62, TABLE_TOP_Z))

    # 8. Small terracotta pot with procedural rosette succulent (right-back corner: x=-0.66, y=-0.76)
    bm_pl = bmesh.new()
    pot = bmesh.new()
    bmesh.ops.create_cone(pot, cap_ends=True, cap_tris=False, segments=28, radius1=0.038, radius2=0.052, depth=0.088)
    geo.add_bm_into(bm_pl, pot, mat_idx=0, xf=Matrix.Translation((0, 0, 0.044)))
    soil = geo.cylinder_bm(0.047, 0.008, segs=24)
    geo.add_bm_into(bm_pl, soil, mat_idx=1, xf=Matrix.Translation((0, 0, 0.084)))
    # Layered organic succulent rosette petals
    for ring_i, (n_pet, r_rad, pitch, sc) in enumerate(((5, 0.010, 0.35, 0.65), (7, 0.020, 0.68, 0.85), (9, 0.030, 0.98, 1.0))):
        for k in range(n_pet):
            ang = 2.0 * math.pi * (k + 0.35 * ring_i) / n_pet
            leaf = geo.rounded_box_bm(0.016 * sc, 0.034 * sc, 0.007 * sc, 0.003 * sc, 3)
            xf = (
                Matrix.Translation((r_rad * math.cos(ang), r_rad * math.sin(ang), 0.095 + 0.006 * (2 - ring_i)))
                @ Matrix.Rotation(ang - math.pi / 2, 4, "Z")
                @ Matrix.Rotation(pitch, 4, "X")
            )
            geo.add_bm_into(bm_pl, leaf, mat_idx=2, xf=xf)
    geo.new_mesh_obj("Prop_Succulent", bm_pl, mats=[mats["terracotta"], mats["optical_black"], mats["foliage"]], coll=coll, loc=(-0.66, -0.76, TABLE_TOP_Z))

    # 9. Ergonomic lab chair behind KAIRO-04 so the robot is realistically seated
    bm_ch = bmesh.new()
    seat = geo.rounded_box_bm(0.44, 0.42, 0.055, 0.018, 3)
    geo.add_bm_into(bm_ch, seat, mat_idx=0, xf=Matrix.Translation((0, 0.05, 0.54)))
    back = geo.rounded_box_bm(0.42, 0.045, 0.44, 0.016, 3)
    geo.add_bm_into(bm_ch, back, mat_idx=0, xf=Matrix.Translation((0, 0.25, 0.82)) @ Matrix.Rotation(-0.10, 4, "X"))
    post = geo.cylinder_bm(0.026, 0.50, segs=20)
    geo.add_bm_into(bm_ch, post, mat_idx=1, xf=Matrix.Translation((0, 0.06, 0.27)))
    geo.new_mesh_obj("Lab_Chair", bm_ch, mats=[mats["chair_fabric"], mats["dark_steel"]], coll=coll)


def _build_architecture_and_background(coll, mats: Dict[str, object]):
    """Build architectural corner robotics studio: real mullioned windows, acoustic slat wall,
    rear test-equipment shelving with glowing oscilloscope, overhead pendant LED & exterior view.
    """
    # Floor & ceiling
    fl = geo.rounded_box_bm(7.0, 7.0, 0.05, 0.0, 1)
    geo.new_mesh_obj("Arch_Floor", fl, mats=[mats["wall"]], coll=coll, loc=(0.0, 0.0, -0.025))
    ceil = geo.rounded_box_bm(7.0, 7.0, 0.05, 0.0, 1)
    geo.new_mesh_obj("Arch_Ceiling", ceil, mats=[mats["wall"]], coll=coll, loc=(0.0, 0.0, 3.05))

    # Rear wall behind KAIRO-04 (at y = +1.95) — rich warm concrete + walnut acoustic slats!
    rear_w = geo.rounded_box_bm(6.5, 0.14, 3.1, 0.0, 1)
    geo.new_mesh_obj("Arch_RearWall", rear_w, mats=[mats["wall"]], coll=coll, loc=(0.0, 2.02, 1.52))

    # Vertical acoustic walnut wood slats on rear wall (x in [-1.8, +0.4]) for warm depth behind robot!
    bm_slats = bmesh.new()
    for i in range(28):
        sx = -1.75 + i * 0.075
        sl = geo.rounded_box_bm(0.038, 0.028, 2.85, 0.003, 1)
        geo.add_bm_into(bm_slats, sl, mat_idx=0, xf=Matrix.Translation((sx, 1.93, 1.48)))
    geo.new_mesh_obj("Arch_AcousticSlats", bm_slats, mats=[mats["wood_slat"]], coll=coll)

    # Right solid concrete wall (at x = -2.45)
    right_w = geo.rounded_box_bm(0.14, 6.2, 3.1, 0.0, 1)
    geo.new_mesh_obj("Arch_RightWall", right_w, mats=[mats["wall"]], coll=coll, loc=(-2.48, 0.0, 1.52))

    # LEFT WALL (at x = +2.15): Real architectural floor-to-ceiling mullioned window wall!
    # Built as lower knee-wall (z < 0.72) + header (z > 2.72) + dark steel mullions + glass
    # leaving HUGE unobstructed glazing in z in [0.72, 2.72], y in [-1.6, +1.7] so sun streams right in!
    knee = geo.rounded_box_bm(0.18, 6.0, 0.72, 0.005, 1)
    geo.new_mesh_obj("Arch_WinKneeWall", knee, mats=[mats["wall"]], coll=coll, loc=(2.18, 0.0, 0.36))
    head = geo.rounded_box_bm(0.18, 6.0, 0.38, 0.005, 1)
    geo.new_mesh_obj("Arch_WinHeader", head, mats=[mats["wall"]], coll=coll, loc=(2.18, 0.0, 2.88))
    bm_mul = bmesh.new()
    for my in (-1.55, -0.50, 0.55, 1.65):
        post = geo.rounded_box_bm(0.085, 0.055, 2.04, 0.004, 2)
        geo.add_bm_into(bm_mul, post, mat_idx=0, xf=Matrix.Translation((2.16, my, 1.72)))
    for mz in (0.74, 1.88, 2.70):
        trans = geo.rounded_box_bm(0.085, 3.30, 0.048, 0.004, 2)
        geo.add_bm_into(bm_mul, trans, mat_idx=0, xf=Matrix.Translation((2.16, 0.05, mz)))
    geo.new_mesh_obj("Arch_WindowMullions", bm_mul, mats=[mats["dark_steel"]], coll=coll)
    # Thin architectural glass pane (shadow rays transparent so sun casts crisp mullion shadows!)
    win_g = geo.rounded_box_bm(0.008, 3.20, 1.94, 0.0, 1)
    geo.new_mesh_obj("Arch_WindowGlass", win_g, mats=[mats["window_glass"]], coll=coll, loc=(2.17, 0.05, 1.72))

    # Exterior view beyond the left window (x > 4.2, strictly LOW z < 1.35 or far off-axis
    # so NOTHING ever intercepts the 21.5 deg sun ray coming into the workbench!)
    ext_parapet = geo.rounded_box_bm(0.45, 12.0, 1.05, 0.02, 2)
    geo.new_mesh_obj("Ext_CourtyardWall", ext_parapet, mats=[mats["wall"]], coll=coll, loc=(5.4, 0.0, 0.52))
    # Sunlit courtyard trees placed off-sun-axis (y = -2.2 and y = -4.2) for rich green bokeh outside window!
    for tx, ty, tr in ((5.8, -2.0, 0.85), (6.5, -3.8, 1.15), (7.2, 2.6, 1.05)):
        bm_tr = bmesh.new()
        trunk = geo.cylinder_bm(0.09, 1.8, segs=12)
        geo.add_bm_into(bm_tr, trunk, mat_idx=0, xf=Matrix.Translation((0, 0, 0.9)))
        for ox, oy, oz, rr in ((0, 0, 2.1, tr), (0.4, 0.3, 1.8, tr * 0.7), (-0.3, -0.3, 1.9, tr * 0.75)):
            sph = bmesh.new()
            bmesh.ops.create_icosphere(sph, subdivisions=2, radius=rr)
            geo.add_bm_into(bm_tr, sph, mat_idx=1, xf=Matrix.Translation((ox, oy, oz)))
        geo.new_mesh_obj(f"Ext_Tree_{ty:+.1f}", bm_tr, mats=[mats["walnut"], mats["foliage"]], coll=coll, loc=(tx, ty, 0.0))

    # Rear Robotics Equipment Shelf & Bench (y = +1.55, x in [-1.5, +1.5]) for layered lab depth!
    bm_rack = bmesh.new()
    for rx in (-1.35, -0.25, 0.85, 1.55):
        for ry in (1.32, 1.78):
            col = geo.rounded_box_bm(0.036, 0.036, 2.10, 0.003, 1)
            geo.add_bm_into(bm_rack, col, mat_idx=0, xf=Matrix.Translation((rx, ry, 1.05)))
    for sz in (0.76, 1.26, 1.74):
        sh = geo.rounded_box_bm(2.96, 0.48, 0.026, 0.003, 1)
        geo.add_bm_into(bm_rack, sh, mat_idx=1, xf=Matrix.Translation((0.10, 1.55, sz)))
    geo.new_mesh_obj("Lab_RearRack", bm_rack, mats=[mats["dark_steel"], mats["walnut"]], coll=coll)

    # Rear Benchtop Digital Oscilloscope with glowing screen (at x = +0.52, y = 1.48, z = 1.28)
    # Visible softly bokeh'd in background of Shots 1, 2, 5, 7!
    sc_body = geo.rounded_box_bm(0.36, 0.22, 0.21, 0.008, 3)
    geo.new_mesh_obj("Lab_Oscilloscope", sc_body, mats=[mats["dark_steel"]], coll=coll, loc=(0.56, 1.50, 1.38), rot=(0, 0, -0.28))
    _uv_plane_obj(
        "Lab_OscilloscopeScreen",
        0.24,
        0.145,
        mats["scope_screen"],
        coll,
        loc=(0.50, 1.385, 1.39),
        rot=(math.pi / 2, 0.0, -0.28),
    )

    # Rear Benchtop DC Power Supply + warm amber & green status LEDs (at x = -0.78, y = 1.48, z = 1.28)
    bm_psu = bmesh.new()
    box = geo.rounded_box_bm(0.32, 0.26, 0.16, 0.006, 2)
    geo.add_bm_into(bm_psu, box, mat_idx=0, xf=Matrix.Translation((0, 0, 0.08)))
    for kx in (-0.09, -0.04, 0.02):
        knob = geo.cylinder_bm(0.014, 0.018, segs=16)
        geo.add_bm_into(bm_psu, knob, mat_idx=1, xf=Matrix.Translation((kx, -0.135, 0.06)) @ Matrix.Rotation(math.pi / 2, 4, "X"))
    disp = geo.rounded_box_bm(0.11, 0.008, 0.042, 0.002, 1)
    geo.add_bm_into(bm_psu, disp, mat_idx=2, xf=Matrix.Translation((0.06, -0.132, 0.11)))
    led_g = geo.rounded_box_bm(0.014, 0.008, 0.014, 0.002, 1)
    geo.add_bm_into(bm_psu, led_g, mat_idx=3, xf=Matrix.Translation((-0.11, -0.132, 0.12)))
    geo.new_mesh_obj(
        "Lab_PowerSupply",
        bm_psu,
        mats=[mats["dark_steel"], mats["aluminum"], mats["led_amber"], mats["led_green"]],
        coll=coll,
        loc=(-0.78, 1.50, 1.275),
        rot=(0, 0, 0.22),
    )

    # Rear Storage Bins on shelves (dark charcoal & matte slate for clean figure-ground pop behind KAIRO-04!)
    for bx, bz, mat_k in ((-1.10, 0.87, "dark_steel"), (-0.52, 0.87, "mug"), (1.18, 0.87, "dark_steel"), (-0.18, 1.36, "dark_steel"), (1.12, 1.35, "mug")):
        bin_bm = geo.rounded_box_bm(0.28, 0.34, 0.18, 0.010, 2)
        geo.new_mesh_obj(f"Lab_Bin_{bx:+.1f}_{bz:.1f}", bin_bm, mats=[mats[mat_k]], coll=coll, loc=(bx, 1.54, bz))

    # Warm recessed LED accent strip under upper shelf (casts soft warm pool onto acoustic slats!)
    strip = geo.rounded_box_bm(2.75, 0.018, 0.012, 0.002, 1)
    geo.new_mesh_obj("Lab_ShelfLEDStrip", strip, mats=[mats["lamp_glow"]], coll=coll, loc=(0.10, 1.66, 1.72))
    l_sh = bpy.data.lights.new("Light_ShelfWarm", type="AREA")
    l_sh.shape = "RECTANGLE"
    l_sh.size = 2.4
    l_sh.size_y = 0.06
    l_sh.energy = 28.0
    l_sh.color = (1.0, 0.72, 0.42)
    o_sh = bpy.data.objects.new("Light_ShelfWarm", l_sh)
    o_sh.location = (0.10, 1.66, 1.70)
    o_sh.rotation_euler = (0.18, 0.0, 0.0)
    geo.link_obj(o_sh, coll)

    # Overhead Linear Architectural Pendant Fixture above the workbench (practical 4800 K key-top!)
    bm_pend = bmesh.new()
    housing = geo.rounded_box_bm(1.48, 0.085, 0.055, 0.006, 2)
    geo.add_bm_into(bm_pend, housing, mat_idx=0)
    diff = geo.rounded_box_bm(1.44, 0.068, 0.008, 0.002, 1)
    geo.add_bm_into(bm_pend, diff, mat_idx=1, xf=Matrix.Translation((0, 0, -0.028)))
    for px in (-0.58, 0.58):
        wire = geo.cylinder_bm(0.0022, 0.90, segs=8)
        geo.add_bm_into(bm_pend, wire, mat_idx=0, xf=Matrix.Translation((px, 0, 0.48)))
    geo.new_mesh_obj("Lab_OverheadPendant", bm_pend, mats=[mats["dark_steel"], mats["overhead_glow"]], coll=coll, loc=(0.0, -0.34, 2.28))

    l_ov = bpy.data.lights.new("Light_OverheadLED", type="AREA")
    l_ov.shape = "RECTANGLE"
    l_ov.size = 1.38
    l_ov.size_y = 0.14
    l_ov.energy = 28.0
    l_ov.color = (0.96, 0.97, 1.0)
    o_ov = bpy.data.objects.new("Light_OverheadLED", l_ov)
    o_ov.location = (0.0, -0.34, 2.23)
    o_ov.rotation_euler = (0.06, 0.0, 0.0)
    geo.link_obj(o_ov, coll)

    # Soft bounce fill card on right side (-X) — INVISIBLE to glossy/camera/transmission
    # so it lifts right-side shadow falloff gently with ZERO fake reflection rectangles!
    l_fill = bpy.data.lights.new("Light_SoftBounceFill", type="AREA")
    l_fill.shape = "RECTANGLE"
    l_fill.size = 1.6
    l_fill.size_y = 1.2
    l_fill.energy = 11.0
    l_fill.color = (0.86, 0.91, 1.0)
    o_fill = bpy.data.objects.new("Light_SoftBounceFill", l_fill)
    o_fill.location = (-1.45, -0.75, 1.35)
    aim_f = (Vector((0.0, -0.32, 0.95)) - o_fill.location).to_track_quat("-Z", "Y")
    o_fill.rotation_euler = aim_f.to_euler()
    o_fill.visible_camera = False
    o_fill.visible_glossy = False
    o_fill.visible_transmission = False
    geo.link_obj(o_fill, coll)


def build_environment(coll, mats: Dict[str, object]):
    """Build the complete sunlit robotics laboratory, workbench props, and lighting."""
    _setup_physical_world()
    _build_workbench_and_props(coll, mats)
    _build_architecture_and_background(coll, mats)
