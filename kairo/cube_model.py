"""True stickerless 57.0 mm magnetic speedcube geometry (GAN / MoYu precision aesthetic).

Key realism upgrades over stickered-block models:
  - Full wrap-around colored ABS shells meeting cleanly along precision chamfered bevels
    (no bulky black sticker borders!).
  - Authentic recessed dark-graphite internal speedcube core & stalks so spinning layers
    reveal real mechanical depth through the rounded Florian corner-cutting gaps.
  - Flush beveled center-cap seam rings on all 6 centers + micro-groove edge seams.
  - Every cubie object's origin sits at the cube center (0,0,0), so layer rotations are
    pure rigid-body quaternions with exact curved motion blur.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import geo
from .config import CUBIE_BEVEL, CUBIE_PITCH, CUBIE_SIZE
from .cube_logic import FACE_COLOR, FACE_NORMAL, CubeState


def _build_cubie_mesh(home: Tuple[int, int, int], core_mat, tile_mats: Dict[str, object]):
    hv = Vector(home) * CUBIE_PITCH
    # Identify which outer cube faces this cubie belongs to
    vis_faces: List[Tuple[str, Vector]] = []
    for f_name, n_arr in FACE_NORMAL.items():
        nv = Vector(n_arr.tolist())
        if Vector(home).dot(nv) > 0.5:
            vis_faces.append((f_name, nv))

    # Material list: index 0 = matte dark internal core; 1..k = colored stickerless ABS shells
    mats = [core_mat]
    face_to_mat_idx: Dict[str, int] = {}
    for f_name, _ in vis_faces:
        mats.append(tile_mats[FACE_COLOR[f_name]])
        face_to_mat_idx[f_name] = len(mats) - 1

    # 1. Precision beveled cubie outer shell (stickerless full-color wrap)
    bm = geo.rounded_box_bm(CUBIE_SIZE, CUBIE_SIZE, CUBIE_SIZE, CUBIE_BEVEL, segs=5)
    bm.faces.ensure_lookup_table()

    # Assign every polygon of the cubie shell to the matching visible face's colored ABS
    # (inward-facing faces pointing into the cube interior remain dark matte core plastic!)
    for f in bm.faces:
        fn = f.normal
        fc = f.calc_center_median()
        best_score = -1e9
        best_mat = 0
        for f_name, nv in vis_faces:
            # Combine normal alignment and spatial side so corner/edge bevels split cleanly at 45 deg
            score = fn.dot(nv) * 0.72 + (fc.dot(nv) / (0.5 * CUBIE_SIZE)) * 0.35
            if score > best_score:
                best_score = score
                best_mat = face_to_mat_idx[f_name]
        # Purely interior sides facing away from all visible outer faces stay dark internal core
        if best_score < 0.08:
            f.material_index = 0
        else:
            f.material_index = best_mat

    # 2. Subtle precision surface detail on flat outer faces (micro-raised glossy tile plateau
    #    & circular center-cap parting groove on the 6 center cubies)
    bm.faces.ensure_lookup_table()
    flat_outer = []
    for f in bm.faces:
        if f.material_index == 0:
            continue
        for f_name, nv in vis_faces:
            if f.normal.dot(nv) > 0.985 and f.calc_area() > 6e-5:
                flat_outer.append((f, f_name, nv, f.material_index))
                break

    if flat_outer:
        # Subtle 0.09 mm micro-step chamfer on each outer facelet giving realistic ABS mold catchlights
        is_center = len(vis_faces) == 1
        for f, f_name, nv, m_idx in flat_outer:
            if not f.is_valid:
                continue
            res1 = bmesh.ops.inset_individual(bm, faces=[f], thickness=0.0011 if is_center else 0.00055, depth=-0.00010)
            for rim in res1["faces"]:
                rim.material_index = m_idx
            res2 = bmesh.ops.inset_individual(bm, faces=[f], thickness=0.00028, depth=0.00014)
            for rim in res2["faces"]:
                rim.material_index = m_idx
            f.material_index = m_idx

    # Translate cubie shell to its home offset relative to cube center (0,0,0)
    bmesh.ops.translate(bm, vec=hv, verts=bm.verts)

    # 3. Internal mechanical torpedo / stalk bridging toward the central core (0,0,0)
    #    so when layers rotate 30-60 deg, gaps reveal real internal dark-ABS speedcube hardware
    stalk_dir = -hv.normalized()
    stalk_len = hv.length * 0.62
    stalk_r = 0.0042 if len(vis_faces) == 1 else 0.0032
    s_bm = geo.cylinder_bm(stalk_r, stalk_len, segs=12, bevel=0.0004, bevel_segs=1)
    q = Vector((0.0, 0.0, 1.0)).rotation_difference(stalk_dir)
    xf = Matrix.Translation(hv * 0.56) @ q.to_matrix().to_4x4()
    geo.add_bm_into(bm, s_bm, mat_idx=0, xf=xf)

    return bm, mats


def build_cube(coll, core_mat, tile_mats: Dict[str, object], name: str = "SpeedCube"):
    """Build the complete 57 mm stickerless 3x3 speedcube under a root Empty."""
    root = bpy.data.objects.new(name, None)
    root.empty_display_size = 0.04
    root.empty_display_type = "CUBE"
    root.rotation_mode = "QUATERNION"
    geo.link_obj(root, coll)

    # Inner spherical precision spider core at (0,0,0) so you never see daylight through gaps
    core_bm = bmesh.new()
    bmesh.ops.create_uvsphere(core_bm, u_segments=20, v_segments=12, radius=0.0128)
    geo.new_mesh_obj("CubeCoreSpider", core_bm, mats=[core_mat], parent=root, coll=coll, smooth=True)

    state = CubeState()
    cubies: List[bpy.types.Object] = []
    for idx, home in enumerate(state.homes):
        bm, mats = _build_cubie_mesh(home, core_mat, tile_mats)
        obj = geo.new_mesh_obj(
            f"Cubie_{home[0]:+d}{home[1]:+d}{home[2]:+d}",
            bm,
            mats=mats,
            parent=root,
            coll=coll,
            smooth=True,
        )
        obj.rotation_mode = "QUATERNION"
        cubies.append(obj)
    return root, cubies
