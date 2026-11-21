"""Procedural hard-surface bmesh geometry primitives for KAIRO-04 and the lab scene."""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import bmesh
import bpy
from mathutils import Matrix, Vector


def ensure_collection(name: str, parent=None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        if parent is None:
            bpy.context.scene.collection.children.link(coll)
        else:
            parent.children.link(coll)
    return coll


def link_obj(obj, coll):
    if obj.name not in coll.objects:
        coll.objects.link(obj)
    return obj


def mesh_from_bm(name: str, bm: bmesh.types.BMesh, smooth: bool = True) -> bpy.types.Mesh:
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    return me


def new_mesh_obj(
    name: str,
    bm: bmesh.types.BMesh,
    mats=None,
    parent=None,
    coll=None,
    smooth: bool = True,
    loc=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0),
) -> bpy.types.Object:
    me = mesh_from_bm(name, bm, smooth=smooth)
    if mats:
        if not isinstance(mats, (list, tuple)):
            mats = [mats]
        for m in mats:
            me.materials.append(m)
    obj = bpy.data.objects.new(name, me)
    obj.location = loc
    obj.rotation_euler = rot
    if parent is not None:
        obj.parent = parent
    if coll is not None:
        link_obj(obj, coll)
    return obj


def add_bm_into(dst: bmesh.types.BMesh, src: bmesh.types.BMesh, mat_idx: Optional[int] = None, xf: Optional[Matrix] = None):
    """Merge `src` bmesh into `dst`, optionally transforming and assigning material index, then free `src`."""
    if xf is not None:
        bmesh.ops.transform(src, matrix=xf, verts=src.verts)
    if mat_idx is not None:
        for f in src.faces:
            f.material_index = mat_idx
    tmp = bpy.data.meshes.new("_tmp_merge")
    src.to_mesh(tmp)
    src.free()
    dst.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)


def rounded_box_bm(sx: float, sy: float, sz: float, bevel: float = 0.002, segs: int = 4) -> bmesh.types.BMesh:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    if bevel > 1e-6:
        max_b = 0.46 * min(sx, sy, sz)
        b = min(bevel, max_b)
        bmesh.ops.bevel(
            bm,
            geom=list(bm.edges),
            offset=b,
            segments=segs,
            profile=0.5,
            affect="EDGES",
            clamp_overlap=True,
        )
    return bm


def cylinder_bm(radius: float, depth: float, segs: int = 32, bevel: float = 0.0, bevel_segs: int = 3) -> bmesh.types.BMesh:
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=segs,
        radius1=radius,
        radius2=radius,
        depth=depth,
    )
    if bevel > 1e-6:
        cap_edges = [e for e in bm.edges if abs(e.verts[0].co.z) > 0.49 * depth and abs(e.verts[1].co.z) > 0.49 * depth]
        bmesh.ops.bevel(
            bm,
            geom=cap_edges,
            offset=min(bevel, radius * 0.45, depth * 0.45),
            segments=bevel_segs,
            profile=0.5,
            affect="EDGES",
            clamp_overlap=True,
        )
    return bm


def loft_rings_bm(rings: Sequence[Sequence[Vector]], cap_start: bool = True, cap_end: bool = True) -> bmesh.types.BMesh:
    """Create a smooth closed lofted tube from a sequence of equal-length 3D point loops."""
    bm = bmesh.new()
    n_ring = len(rings[0])
    v_rings = []
    for ring in rings:
        v_rings.append([bm.verts.new(p) for p in ring])
    for r0, r1 in zip(v_rings[:-1], v_rings[1:]):
        for i in range(n_ring):
            j = (i + 1) % n_ring
            bm.faces.new((r0[i], r0[j], r1[j], r1[i]))
    if cap_start:
        c0 = bm.verts.new(sum((v.co for v in v_rings[0]), Vector()) / n_ring)
        for i in range(n_ring):
            j = (i + 1) % n_ring
            bm.faces.new((v_rings[0][j], v_rings[0][i], c0))
    if cap_end:
        c1 = bm.verts.new(sum((v.co for v in v_rings[-1]), Vector()) / n_ring)
        for i in range(n_ring):
            j = (i + 1) % n_ring
            bm.faces.new((v_rings[-1][i], v_rings[-1][j], c1))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def superellipse_ring(
    rx: float,
    rz: float,
    y: float,
    power: float = 2.6,
    n: int = 28,
    z_off: float = 0.0,
    x_off: float = 0.0,
    flat_bottom: float = 1.0,
) -> List[Vector]:
    """Generate a sculpted superellipse cross-section ring in the XZ plane at `y`."""
    pts: List[Vector] = []
    exp = 2.0 / power
    for i in range(n):
        a = 2.0 * math.pi * i / n
        ca, sa = math.cos(a), math.sin(a)
        x = math.copysign(abs(ca) ** exp, ca) * rx + x_off
        z = math.copysign(abs(sa) ** exp, sa) * rz
        if z < 0.0:
            z *= flat_bottom
        pts.append(Vector((x, y, z + z_off)))
    return pts


def tube_along_points_bm(pts: Sequence[Vector], radius: float, segs: int = 12) -> bmesh.types.BMesh:
    """Sweep a circular cable/conduit smoothly along a 3D polyline."""
    rings = []
    n_pts = len(pts)
    prev_n = None
    for i, p in enumerate(pts):
        if i == 0:
            tang = (pts[1] - pts[0]).normalized()
        elif i == n_pts - 1:
            tang = (pts[-1] - pts[-2]).normalized()
        else:
            tang = (pts[i + 1] - pts[i - 1]).normalized()
        if prev_n is None:
            up = Vector((0, 0, 1)) if abs(tang.z) < 0.9 else Vector((1, 0, 0))
            norm = tang.cross(up).normalized()
        else:
            norm = (prev_n - tang * prev_n.dot(tang))
            norm = norm.normalized() if norm.length > 1e-6 else tang.orthogonal().normalized()
        binorm = tang.cross(norm).normalized()
        prev_n = norm
        ring = [
            p + (norm * math.cos(2.0 * math.pi * k / segs) + binorm * math.sin(2.0 * math.pi * k / segs)) * radius
            for k in range(segs)
        ]
        rings.append(ring)
    return loft_rings_bm(rings, cap_start=True, cap_end=True)


def hex_socket_bolt_bm(head_r: float = 0.0024, head_h: float = 0.0012) -> bmesh.types.BMesh:
    """Machined titanium M3/M4 socket-head cap screw oriented with axis along +Z."""
    bm = cylinder_bm(head_r, head_h, segs=16, bevel=head_r * 0.16, bevel_segs=2)
    bmesh.ops.translate(bm, vec=(0.0, 0.0, 0.5 * head_h), verts=bm.verts)
    # Recessed hex socket ring on top face
    top_faces = [f for f in bm.faces if f.normal.z > 0.95 and f.calc_center_median().z > 0.8 * head_h]
    if top_faces:
        res = bmesh.ops.inset_individual(bm, faces=top_faces, thickness=head_r * 0.46, depth=0.0)
        bmesh.ops.translate(bm, vec=(0.0, 0.0, -head_h * 0.55), verts=list({v for f in top_faces for v in f.verts}))
    return bm
