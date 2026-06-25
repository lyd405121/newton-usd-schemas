#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate AOUSD curve-deformable rod/cable assets from a USD file.

Usage:
    python simulate_rod.py <usd_file>

Examples:
    python simulate_rod.py asset/LC_SC_S.usda
    python simulate_rod.py asset/test_graph_attach.usda
    python simulate_rod.py Graph.usda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Remove REPO_ROOT from sys.path if present — it contains a newton/ directory
# that would shadow the conda-installed newton package.
sys.path = [p for p in sys.path if Path(p).resolve() != REPO_ROOT]

import newton
import newton.solvers
import newton.viewer
import warp as wp
from pxr import Usd, UsdGeom

import newton_usd_schemas  # noqa: F401 - registers Newton extension schemas

DEFAULT_STEPS = 200
DEFAULT_SUBSTEPS = 10
DEFAULT_ITERATIONS = 5
DEFAULT_FPS = 120


def _resolve_usd_path(path_arg: str) -> Path:
    """Resolve a USD path from cwd, script dir, repo root, or repo asset dir."""
    raw_path = Path(path_arg).expanduser()

    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                Path.cwd() / raw_path,
                SCRIPT_DIR / raw_path,
                REPO_ROOT / raw_path,
                REPO_ROOT / "asset" / raw_path,
            ]
        )

    candidates = list(dict.fromkeys(candidates))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"USD file not found: {path_arg}\nChecked:\n  - {checked}"
    )


# ---------------------------------------------------------------------------
# USD parsing helpers
# ---------------------------------------------------------------------------

def _has_api(prim: Usd.Prim, api_name: str) -> bool:
    """Return true for both registered and proposal/unknown applied API tokens."""
    if prim.HasAPI(api_name) or api_name in prim.GetAppliedSchemas():
        return True
    api_schemas = prim.GetMetadata("apiSchemas")
    return bool(api_schemas and api_schemas.HasItem(api_name))


def _float_attr(prim: Usd.Prim, name: str, default: float | None = None) -> float | None:
    attr = prim.GetAttribute(name)
    if attr and attr.HasAuthoredValue():
        value = attr.Get()
        if value is not None:
            return float(value)
    return default


def _find_bound_material_prim(prim: Usd.Prim) -> Usd.Prim | None:
    """Find a directly or ancestrally bound physics/material prim."""
    current = prim
    while current and current.IsValid():
        rel = current.GetRelationship("material:binding:physics")
        if not rel or not rel.HasAuthoredTargets():
            rel = current.GetRelationship("material:binding")
        if rel and rel.HasAuthoredTargets():
            targets = rel.GetTargets()
            if targets:
                mat = prim.GetStage().GetPrimAtPath(targets[0])
                if mat and mat.IsValid():
                    return mat
        current = current.GetParent()
    return None


def _material_values(mat: Usd.Prim | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if mat is None:
        return out
    for name in (
        "physics:thickness",
        "physics:stretchStiffness",
        "physics:bendStiffness",
        "newton:stretchDamping",
        "newton:bendDamping",
    ):
        val = _float_attr(mat, name)
        if val is not None:
            out[name] = val
    return out


def _curve_material(prim: Usd.Prim) -> dict[str, float]:
    return _material_values(_find_bound_material_prim(prim))


def _curve_segment_materials(
    prim: Usd.Prim,
    segment_count: int,
    base_material: dict[str, float],
) -> list[dict[str, float]]:
    """Resolve per-segment material values from child GeomSubset bindings."""
    materials = [dict(base_material) for _ in range(segment_count)]
    if segment_count <= 0:
        return materials

    for child in prim.GetChildren():
        if child.GetTypeName() != "GeomSubset":
            continue
        if child.GetAttribute("elementType").Get() != "segment":
            continue
        indices = child.GetAttribute("indices").Get() or []
        subset_material = _curve_material(child)
        if not subset_material:
            continue
        for idx in indices:
            segment_idx = int(idx)
            if 0 <= segment_idx < segment_count:
                materials[segment_idx] = dict(subset_material)
            else:
                print(f"[warn] {child.GetPath()}: segment index {segment_idx} outside 0..{segment_count - 1}")
    return materials


def _ancestor_has_api(prim: Usd.Prim, api_name: str) -> bool:
    current = prim
    while current and current.IsValid():
        if _has_api(current, api_name):
            return True
        current = current.GetParent()
    return False


def _curve_points(curves: UsdGeom.BasisCurves) -> list[wp.vec3]:
    pts = curves.GetPointsAttr().Get()
    if not pts:
        return []
    from pxr import Gf, Usd as _Usd
    xform = UsdGeom.Xformable(curves)
    mat = xform.ComputeLocalToWorldTransform(_Usd.TimeCode.Default())
    result = []
    for p in pts:
        pw = mat.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        result.append(wp.vec3(float(pw[0]), float(pw[1]), float(pw[2])))
    return result


def _curve_vertex_counts(curves: UsdGeom.BasisCurves, n_points: int) -> list[int]:
    counts = curves.GetCurveVertexCountsAttr().Get()
    return [int(c) for c in counts] if counts else [n_points]


def _curve_radius(curves: UsdGeom.BasisCurves, material: dict[str, float]) -> float:
    return _segment_radius(curves, 0, material, 0.005)


def _segment_radius(
    curves: UsdGeom.BasisCurves,
    segment_index: int,
    material: dict[str, float],
    fallback: float,
) -> float:
    thickness = material.get("physics:thickness")
    if thickness is not None and thickness > 0.0:
        return 0.5 * thickness
    widths = curves.GetWidthsAttr().Get()
    if widths and len(widths) > 0:
        width_idx = min(max(segment_index, 0), len(widths) - 1)
        return max(float(widths[width_idx]) * 0.5, 1.0e-6)
    return fallback


def _average_material(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in set(left) | set(right):
        a = left.get(key)
        b = right.get(key)
        if a is not None and b is not None:
            out[key] = 0.5 * (a + b)
        elif a is not None:
            out[key] = a
        elif b is not None:
            out[key] = b
    return out


def _format_range(values: list[float]) -> str:
    if not values:
        return "none"
    return f"{min(values):.4g}..{max(values):.4g}"


def _apply_segment_material_overrides(
    builder: newton.ModelBuilder,
    body_ids: list[int],
    joint_ids: list[int],
    segment_materials: list[dict[str, float]],
    segment_radii: list[float],
    label: str,
) -> list[float]:
    """Apply GeomSubset segment material values after add_rod/add_rod_graph."""
    if not body_ids or not segment_radii:
        return []

    edge_radii = [segment_radii[min(i, len(segment_radii) - 1)] for i in range(len(body_ids))]

    for edge_idx, body_id in enumerate(body_ids):
        radius = edge_radii[edge_idx]
        for shape_idx in builder.body_shapes.get(body_id, []):
            scale = builder.shape_scale[shape_idx]
            half_height = float(scale[1])
            builder.shape_scale[shape_idx] = (radius, half_height, radius)
            builder.shape_collision_radius[shape_idx] = radius + abs(half_height)

    for joint_order, joint_idx in enumerate(joint_ids):
        if joint_idx < 0 or joint_idx >= len(builder.joint_qd_start):
            continue
        left = segment_materials[min(joint_order, len(segment_materials) - 1)] if segment_materials else {}
        right = segment_materials[min(joint_order + 1, len(segment_materials) - 1)] if segment_materials else left
        material = _average_material(left, right)
        dof_start = builder.joint_qd_start[joint_idx]
        radius = edge_radii[min(joint_order, len(edge_radii) - 1)]
        # Compute segment length from shape scale (half_height * 2)
        body_id = body_ids[min(joint_order, len(body_ids) - 1)]
        seg_len = 0.1  # fallback
        for shape_idx in builder.body_shapes.get(body_id, []):
            seg_len = float(builder.shape_scale[shape_idx][1]) * 2.0
            break
        from newton._src.utils.cable import create_cable_stiffness_from_elastic_moduli
        if dof_start < len(builder.joint_target_ke):
            raw_stretch = material.get("physics:stretchStiffness")
            if raw_stretch is not None:
                builder.joint_target_ke[dof_start] = create_cable_stiffness_from_elastic_moduli(raw_stretch, radius, seg_len)[0]
            builder.joint_target_kd[dof_start] = material.get(
                "newton:stretchDamping", builder.joint_target_kd[dof_start]
            )
        if dof_start + 1 < len(builder.joint_target_ke):
            raw_bend = material.get("physics:bendStiffness")
            if raw_bend is not None:
                builder.joint_target_ke[dof_start + 1] = create_cable_stiffness_from_elastic_moduli(raw_bend, radius, seg_len)[1]
            builder.joint_target_kd[dof_start + 1] = material.get(
                "newton:bendDamping", builder.joint_target_kd[dof_start + 1]
            )

    stretch_values = [m["physics:stretchStiffness"] for m in segment_materials if "physics:stretchStiffness" in m]
    bend_values = [m["physics:bendStiffness"] for m in segment_materials if "physics:bendStiffness" in m]
    if len({round(r, 12) for r in edge_radii}) > 1 or len({round(v, 12) for v in stretch_values}) > 1:
        print(
            f"[subset] {label}: applied segment overrides "
            f"radius={_format_range(edge_radii)} stretch={_format_range(stretch_values)} bend={_format_range(bend_values)}"
        )
    return edge_radii


def _wrap_in_articulation(prim: Usd.Prim) -> bool:
    # Only wrap when an articulation root is explicitly present in the
    # deformable hierarchy.
    return _ancestor_has_api(prim, "NewtonArticulationRootAPI") or _ancestor_has_api(prim, "PhysicsArticulationRootAPI")


def parse_rods(stage: Usd.Stage, builder: newton.ModelBuilder) -> dict:
    """Parse AOUSD curve deformables into Newton rods or rod graphs.

    Curve-to-curve `PhysicsAttachment` prims are treated as topology: attached
    point pairs are welded into a single graph component and imported with
    `ModelBuilder.add_rod_graph()`. World/body attachments are parsed later.
    """
    from collections import defaultdict

    curves_by_path: dict[str, dict] = {}

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.BasisCurves):
            continue
        if not _has_api(prim, "PhysicsCurvesDeformableSimAPI"):
            continue

        path = str(prim.GetPath())
        curves = UsdGeom.BasisCurves(prim)
        all_positions = _curve_points(curves)
        counts = _curve_vertex_counts(curves, len(all_positions))
        material = _curve_material(prim)
        radius = _curve_radius(curves, material)
        closed = curves.GetWrapAttr().Get() == UsdGeom.Tokens.periodic
        wrap = _wrap_in_articulation(prim)
        segment_counts = [count if closed else max(count - 1, 0) for count in counts]
        all_segment_materials = _curve_segment_materials(prim, sum(segment_counts), material)
        all_segment_radii = [
            _segment_radius(curves, segment_idx, segment_material, radius)
            for segment_idx, segment_material in enumerate(all_segment_materials)
        ]

        offset = 0
        segment_offset = 0
        for curve_index, count in enumerate(counts):
            positions = all_positions[offset : offset + count]
            segment_count = segment_counts[curve_index]
            segment_materials = all_segment_materials[segment_offset : segment_offset + segment_count]
            segment_radii = all_segment_radii[segment_offset : segment_offset + segment_count]
            offset += count
            segment_offset += segment_count
            if len(positions) < 2:
                print(f"[warn] {path}[{curve_index}]: need at least 2 curve points, skipping")
                continue
            key = path if len(counts) == 1 else f"{path}[{curve_index}]"
            curves_by_path[key] = {
                "key": key,
                "curve_path": path,
                "prim": prim,
                "curve_index": curve_index,
                "positions": positions,
                "material": material,
                "segment_materials": segment_materials,
                "segment_radii": segment_radii,
                "radius": segment_radii[0] if segment_radii else radius,
                "closed": closed,
                "wrap": wrap,
            }

    parent: dict[str, str] = {key: key for key in curves_by_path}

    def find(key: str) -> str:
        root = parent[key]
        if root != key:
            parent[key] = find(root)
        return parent[key]

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    graph_attachments: list[tuple[str, int, str, int]] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsAttachment":
            continue
        src0_targets = prim.GetRelationship("physics:src0").GetTargets()
        src1_targets = prim.GetRelationship("physics:src1").GetTargets()
        if not src0_targets or not src1_targets:
            continue
        src0 = str(src0_targets[0])
        src1 = str(src1_targets[0])
        if src0 not in curves_by_path or src1 not in curves_by_path:
            continue
        if src0 == src1:
            continue
        indices0 = prim.GetAttribute("physics:indices0").Get() or []
        indices1 = prim.GetAttribute("physics:indices1").Get() or []
        if not indices0 or not indices1:
            continue
        union(src0, src1)
        for i, idx0 in enumerate(indices0):
            idx1 = indices1[i] if i < len(indices1) else indices1[0]
            graph_attachments.append((src0, int(idx0), src1, int(idx1)))

    components: dict[str, list[str]] = defaultdict(list)
    for key in curves_by_path:
        components[find(key)].append(key)

    attachments_by_component: dict[str, list[tuple[str, int, str, int]]] = defaultdict(list)
    for attachment in graph_attachments:
        attachments_by_component[find(attachment[0])].append(attachment)

    rod_info: dict = {}

    for component_key, keys in components.items():
        keys = sorted(keys)
        component_attachments = attachments_by_component.get(component_key, [])
        if len(keys) == 1 and not component_attachments:
            rec = curves_by_path[keys[0]]
            positions = rec["positions"]
            if len(positions) < 3:
                print(f"[warn] {rec['key']}: add_rod requires at least 3 points, skipping")
                continue
            material = rec["material"]
            # Convert elastic moduli to joint stiffness using cross-section geometry.
            # physics:stretchStiffness and bendStiffness are material moduli [Pa],
            # not joint target_ke directly.
            radius = rec["radius"]
            seg_len = sum(
                float(wp.length(positions[i+1] - positions[i]))
                for i in range(len(positions)-1)
            ) / max(len(positions)-1, 1)
            from newton._src.utils.cable import create_cable_stiffness_from_elastic_moduli
            raw_stretch = material.get("physics:stretchStiffness", 1.0e5)
            raw_bend = material.get("physics:bendStiffness", 0.0)
            stretch_stiffness = create_cable_stiffness_from_elastic_moduli(raw_stretch, radius, seg_len)[0] if raw_stretch else None
            bend_stiffness = create_cable_stiffness_from_elastic_moduli(raw_bend, radius, seg_len)[1] if raw_bend else None
            stretch_damping = material.get("newton:stretchDamping", 0.0)
            bend_damping = material.get("newton:bendDamping", 0.0)
            label = rec["key"].replace("/", "_")
            print(
                f"[curve] {rec['key']}: add_rod points={len(positions)} segments={len(positions)-1} "
                f"radius={rec['radius']:.4f} closed={rec['closed']} articulation={rec['wrap']}"
            )
            body_ids, joint_ids = builder.add_rod(
                positions=positions,
                radius=rec["radius"],
                cfg=newton.ModelBuilder.ShapeConfig(),
                stretch_stiffness=stretch_stiffness,
                stretch_damping=stretch_damping,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                closed=rec["closed"],
                wrap_in_articulation=rec["wrap"],
                label=label,
            )
            edge_radii = _apply_segment_material_overrides(
                builder,
                body_ids,
                joint_ids,
                rec["segment_materials"],
                rec["segment_radii"],
                rec["key"],
            )
            edges = [(i, i + 1) for i in range(len(positions) - 1)]
            for edge_idx, body_a in enumerate(body_ids):
                neighbor_indices = [edge_idx - 1, edge_idx + 1]
                if rec["closed"]:
                    if edge_idx == 0:
                        neighbor_indices.append(len(body_ids) - 1)
                    elif edge_idx == len(body_ids) - 1:
                        neighbor_indices.append(0)
                for neighbor_idx in neighbor_indices:
                    if neighbor_idx <= edge_idx or neighbor_idx >= len(body_ids):
                        continue
                    body_b = body_ids[neighbor_idx]
                    for shape_a in builder.body_shapes.get(body_a, []):
                        for shape_b in builder.body_shapes.get(body_b, []):
                            builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))
            half_lengths = [float(wp.length(positions[v] - positions[u])) / 2.0 for u, v in edges]
            node_to_bodies = defaultdict(list)
            for edge_idx, (u, v) in enumerate(edges):
                if edge_idx < len(body_ids):
                    node_to_bodies[u].append(body_ids[edge_idx])
                    node_to_bodies[v].append(body_ids[edge_idx])
            segment_to_body = {i: body_id for i, body_id in enumerate(body_ids)}
            rod_info[rec["key"]] = {
                "curve_path": rec["curve_path"],
                "body_indices": body_ids,
                "joint_indices": joint_ids,
                "wrap": rec["wrap"],
                "positions": positions,
                "half_lengths": half_lengths,
                "edges_list": edges,
                "radii": edge_radii or [rec["radius"]] * len(body_ids),
                "prim": rec["prim"],
                "curve_index": rec["curve_index"],
                "node_to_bodies": dict(node_to_bodies),
                "segment_to_body": segment_to_body,
            }
            continue

        node_parent: dict[tuple[str, int], tuple[str, int]] = {}

        def node_find(node: tuple[str, int]) -> tuple[str, int]:
            node_parent.setdefault(node, node)
            root = node_parent[node]
            if root != node:
                node_parent[node] = node_find(root)
            return node_parent[node]

        def node_union(a: tuple[str, int], b: tuple[str, int]) -> None:
            ra = node_find(a)
            rb = node_find(b)
            if ra != rb:
                node_parent[rb] = ra

        for key in keys:
            rec = curves_by_path[key]
            for i in range(len(rec["positions"])):
                node_find((key, i))
        for src0, idx0, src1, idx1 in component_attachments:
            node_union((src0, idx0), (src1, idx1))

        global_node_for_root: dict[tuple[str, int], int] = {}
        node_positions: list[wp.vec3] = []

        def global_node(local_node: tuple[str, int]) -> int:
            root = node_find(local_node)
            if root not in global_node_for_root:
                key, idx = root
                global_node_for_root[root] = len(node_positions)
                node_positions.append(curves_by_path[key]["positions"][idx])
            return global_node_for_root[root]

        graph_edges: list[tuple[int, int]] = []
        graph_edge_records: list[tuple[str, int, int]] = []
        for key in keys:
            rec = curves_by_path[key]
            positions = rec["positions"]
            local_edges = [(i, i + 1) for i in range(len(positions) - 1)]
            if rec["closed"]:
                local_edges.append((len(positions) - 1, 0))
            for u, v in local_edges:
                gu = global_node((key, u))
                gv = global_node((key, v))
                if gu == gv:
                    print(f"[warn] {key}: skipping zero-length welded graph edge {u}->{v}")
                    continue
                graph_edges.append((gu, gv))
                graph_edge_records.append((key, u, v))

        if not graph_edges:
            print(f"[warn] graph component {component_key}: no graph edges, skipping")
            continue

        first = curves_by_path[keys[0]]
        material = first["material"]
        radius = first["radius"]
        seg_len = sum(
            float(wp.length(node_positions[v] - node_positions[u]))
            for u, v in graph_edges
        ) / max(len(graph_edges), 1)
        from newton._src.utils.cable import create_cable_stiffness_from_elastic_moduli
        raw_stretch = material.get("physics:stretchStiffness", 1.0e5)
        raw_bend = material.get("physics:bendStiffness", 0.0)
        stretch_stiffness = create_cable_stiffness_from_elastic_moduli(raw_stretch, radius, seg_len)[0] if raw_stretch else None
        bend_stiffness = create_cable_stiffness_from_elastic_moduli(raw_bend, radius, seg_len)[1] if raw_bend else None
        stretch_damping = material.get("newton:stretchDamping", 0.0)
        bend_damping = material.get("newton:bendDamping", 0.0)
        wrap = any(curves_by_path[key]["wrap"] for key in keys)
        label = component_key.replace("/", "_")
        print(
            f"[graph] {component_key}: add_rod_graph curves={len(keys)} nodes={len(node_positions)} "
            f"edges={len(graph_edges)} attachments={len(component_attachments)} radius={first['radius']:.4f} articulation={wrap}"
        )
        body_ids, joint_ids = builder.add_rod_graph(
            node_positions=node_positions,
            edges=graph_edges,
            radius=first["radius"],
            cfg=newton.ModelBuilder.ShapeConfig(),
            stretch_stiffness=stretch_stiffness,
            stretch_damping=stretch_damping,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            wrap_in_articulation=wrap,
            label=label,
        )

        graph_segment_materials: list[dict[str, float]] = []
        graph_segment_radii: list[float] = []
        for key, u, v in graph_edge_records:
            rec = curves_by_path[key]
            segment_idx = u if v == u + 1 else len(rec["segment_materials"]) - 1
            segment_idx = min(max(segment_idx, 0), max(len(rec["segment_materials"]) - 1, 0))
            graph_segment_materials.append(
                rec["segment_materials"][segment_idx] if rec["segment_materials"] else rec["material"]
            )
            graph_segment_radii.append(rec["segment_radii"][segment_idx] if rec["segment_radii"] else rec["radius"])
        graph_edge_radii = _apply_segment_material_overrides(
            builder,
            body_ids,
            joint_ids,
            graph_segment_materials,
            graph_segment_radii,
            component_key,
        )

        per_curve_edges: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
        node_to_bodies_by_curve: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        segment_to_body_by_curve: dict[str, dict[int, int]] = defaultdict(dict)
        for edge_idx, (key, u, v) in enumerate(graph_edge_records):
            if edge_idx >= len(body_ids):
                continue
            body_id = body_ids[edge_idx]
            per_curve_edges[key].append((u, v, body_id, edge_idx))
            node_to_bodies_by_curve[key][u].append(body_id)
            node_to_bodies_by_curve[key][v].append(body_id)
            segment_idx = u if v == u + 1 else len(curves_by_path[key]["positions"]) - 1
            segment_to_body_by_curve[key][segment_idx] = body_id

        for key in keys:
            rec = curves_by_path[key]
            curve_edges = per_curve_edges.get(key, [])
            curve_body_ids = [body_id for _, _, body_id, _ in curve_edges]
            edges = [(u, v) for u, v, _, _ in curve_edges]
            edge_indices = [edge_idx for _, _, _, edge_idx in curve_edges]
            positions = rec["positions"]
            half_lengths = [float(wp.length(positions[v] - positions[u])) / 2.0 for u, v in edges]
            rod_info[key] = {
                "curve_path": rec["curve_path"],
                "body_indices": curve_body_ids,
                "joint_indices": joint_ids,
                "wrap": wrap,
                "positions": positions,
                "half_lengths": half_lengths,
                "edges_list": edges,
                "radii": (
                    [graph_edge_radii[i] for i in edge_indices]
                    if graph_edge_radii
                    else [rec["radius"]] * len(curve_body_ids)
                ),
                "prim": rec["prim"],
                "curve_index": rec["curve_index"],
                "node_to_bodies": {idx: bodies for idx, bodies in node_to_bodies_by_curve[key].items()},
                "segment_to_body": dict(segment_to_body_by_curve[key]),
            }

    return rod_info


def _rod_for_curve_path(rod_info: dict, curve_path: str):
    exact = rod_info.get(curve_path)
    if exact is not None:
        return exact
    for info in rod_info.values():
        if info.get("curve_path") == curve_path:
            return info
    return None


def _filter_enabled(prim: Usd.Prim) -> bool:
    attr = prim.GetAttribute("physics:filterEnabled")
    if not attr or not attr.HasAuthoredValue():
        return True
    return bool(attr.Get())


def _filter_element_indices(prim: Usd.Prim, suffix: str) -> list[int] | None:
    counts_attr = prim.GetAttribute(f"physics:groupElemCounts{suffix}")
    indices_attr = prim.GetAttribute(f"physics:groupElemIndices{suffix}")
    counts = counts_attr.Get() if counts_attr and counts_attr.HasAuthoredValue() else []
    indices = indices_attr.Get() if indices_attr and indices_attr.HasAuthoredValue() else []
    if not counts and not indices:
        return None
    if counts and sum(int(c) for c in counts) != len(indices):
        print(
            f"[warn] {prim.GetPath()}: groupElemCounts{suffix} does not match "
            f"groupElemIndices{suffix}, using authored indices directly"
        )
    return [int(i) for i in indices]


def _rigid_shape_ids_for_path(
    stage: Usd.Stage,
    builder: newton.ModelBuilder,
    path: str,
    body_index_map: dict,
    path_shape_map: dict,
) -> list[int]:
    shape_ids: list[int] = []
    if path in path_shape_map:
        shape_ids.append(int(path_shape_map[path]))
    body_id = body_index_map.get(path)
    if body_id is not None:
        shape_ids.extend(int(s) for s in builder.body_shapes.get(body_id, []))

    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        prefix = path + "/"
        for shape_path, shape_id in path_shape_map.items():
            if shape_path.startswith(prefix):
                shape_ids.append(int(shape_id))

    return list(dict.fromkeys(shape_ids))


def _filter_shape_ids_for_path(
    stage: Usd.Stage,
    builder: newton.ModelBuilder,
    rod_info: dict,
    body_index_map: dict,
    path_shape_map: dict,
    path: str,
    element_indices: list[int] | None,
) -> list[int]:
    info = _rod_for_curve_path(rod_info, path)
    if info is not None:
        if element_indices is None:
            element_indices = list(range(len(info["body_indices"])))
        shape_ids: list[int] = []
        segment_to_body = info.get("segment_to_body", {})
        for segment_idx in element_indices:
            body_id = segment_to_body.get(segment_idx)
            if body_id is None and 0 <= segment_idx < len(info["body_indices"]):
                body_id = info["body_indices"][segment_idx]
            if body_id is None:
                print(f"[warn] filter source {path}: segment {segment_idx} was not imported")
                continue
            shape_ids.extend(int(s) for s in builder.body_shapes.get(body_id, []))
        return list(dict.fromkeys(shape_ids))

    if element_indices is not None:
        print(f"[warn] filter source {path}: element indices are only implemented for curves")
    return _rigid_shape_ids_for_path(stage, builder, path, body_index_map, path_shape_map)


def parse_element_collision_filters(
    stage: Usd.Stage,
    builder: newton.ModelBuilder,
    rod_info: dict,
    body_index_map: dict,
    path_shape_map: dict,
) -> None:
    """Parse AOUSD PhysicsElementCollisionFilter into Newton shape filters."""
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsElementCollisionFilter":
            continue
        if not _filter_enabled(prim):
            continue
        src0_targets = prim.GetRelationship("physics:src0").GetTargets()
        src1_targets = prim.GetRelationship("physics:src1").GetTargets()
        if not src0_targets or not src1_targets:
            print(f"[warn] {prim.GetPath()}: missing collision filter sources, skipping")
            continue
        src0 = str(src0_targets[0])
        src1 = str(src1_targets[0])
        elems0 = _filter_element_indices(prim, "0")
        elems1 = _filter_element_indices(prim, "1")
        shapes0 = _filter_shape_ids_for_path(stage, builder, rod_info, body_index_map, path_shape_map, src0, elems0)
        shapes1 = _filter_shape_ids_for_path(stage, builder, rod_info, body_index_map, path_shape_map, src1, elems1)
        if not shapes0 or not shapes1:
            print(f"[warn] {prim.GetPath()}: resolved empty collision filter shape set, skipping")
            continue
        pair_count = 0
        for shape0 in shapes0:
            for shape1 in shapes1:
                if shape0 == shape1:
                    continue
                builder.add_shape_collision_filter_pair(int(shape0), int(shape1))
                pair_count += 1
        elem_desc0 = "all" if elems0 is None else elems0
        elem_desc1 = "all" if elems1 is None else elems1
        print(
            f"[filter] {prim.GetName()}: {src0} segments={elem_desc0} "
            f"<-> {src1} elements={elem_desc1} pairs={pair_count}"
        )

def _pin_point(info: dict, node_idx: int, builder: newton.ModelBuilder) -> None:
    body_ids = info["body_indices"]
    if not body_ids:
        return
    pinned = set(info.get("node_to_bodies", {}).get(node_idx, []))
    if not pinned:
        pinned.add(body_ids[min(max(node_idx, 0), len(body_ids) - 1)])
    try:
        from newton._src.sim.enums import BodyFlags
        kinematic_flag = int(BodyFlags.KINEMATIC)
    except Exception:
        kinematic_flag = None
    for b in pinned:
        builder.body_mass[b] = 0.0
        builder.body_inv_mass[b] = 0.0
        builder.body_inertia[b] = wp.mat33(0.0)
        builder.body_inv_inertia[b] = wp.mat33(0.0)
        if kinematic_flag is not None and hasattr(builder, "body_flags"):
            builder.body_flags[b] = kinematic_flag


def sync_visual_curves(stage: Usd.Stage, rod_info: dict, body_q_np) -> None:
    """Write simulated curve point positions back to the BasisCurves prims."""
    import numpy as np
    from pxr import Vt

    def quat_rotate_z(q, half_len):
        qx, qy, qz, qw = q
        z = half_len
        return np.array(
            [
                2.0 * (qx * qz + qw * qy) * z,
                2.0 * (qy * qz - qw * qx) * z,
                (1.0 - 2.0 * (qx * qx + qy * qy)) * z,
            ],
            dtype=np.float64,
        )

    for info in rod_info.values():
        body_ids = info["body_indices"]
        prim = info["prim"]
        n_nodes = len(info["positions"])
        node_pos = np.zeros((n_nodes, 3), dtype=np.float64)
        node_set = np.zeros(n_nodes, dtype=bool)
        node_radius = np.full(n_nodes, 0.005, dtype=np.float64)

        for e_idx, (u, v) in enumerate(info["edges_list"]):
            if e_idx >= len(body_ids):
                continue
            bid = body_ids[e_idx]
            if bid >= len(body_q_np):
                continue
            bq = body_q_np[bid]
            pos = bq[:3]
            offset = quat_rotate_z(bq[3:], info["half_lengths"][e_idx])
            if not node_set[u]:
                node_pos[u] = pos - offset
                node_set[u] = True
            if not node_set[v]:
                node_pos[v] = pos + offset
                node_set[v] = True
            r = info["radii"][e_idx] if e_idx < len(info["radii"]) else info["radii"][-1]
            node_radius[u] = r
            node_radius[v] = r

        prim.GetAttribute("points").Set(
            Vt.Vec3fArray([(float(p[0]), float(p[1]), float(p[2])) for p in node_pos])
        )
        if prim.GetAttribute("widths"):
            prim.GetAttribute("widths").Set(Vt.FloatArray([float(2.0 * r) for r in node_radius]))


def _node_body(info: dict, node_idx: int) -> int | None:
    body_ids = info["body_indices"]
    if not body_ids:
        return None
    node_bodies = info.get("node_to_bodies", {}).get(node_idx)
    if node_bodies:
        return node_bodies[0]
    return body_ids[min(max(node_idx, 0), len(body_ids) - 1)]


def _node_local_pos(info: dict, builder: newton.ModelBuilder, body_id: int, node_idx: int) -> wp.vec3:
    positions = info["positions"]
    node_idx = min(max(node_idx, 0), len(positions) - 1)
    node_world = positions[node_idx]
    body_world_xform = builder.body_q[body_id]
    return wp.transform_point(wp.transform_inverse(body_world_xform), node_world)


def parse_attachments(stage: Usd.Stage, builder: newton.ModelBuilder, rod_info: dict, body_index_map: dict) -> None:
    """Parse AOUSD PhysicsAttachment prims for curve pins, graph junctions, and bodies."""
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsAttachment":
            continue
        src0_targets = prim.GetRelationship("physics:src0").GetTargets()
        src1_targets = prim.GetRelationship("physics:src1").GetTargets()
        if not src0_targets or not src1_targets:
            print(f"[warn] {prim.GetPath()}: missing attachment sources, skipping")
            continue
        src0 = str(src0_targets[0])
        src1 = str(src1_targets[0])
        info = _rod_for_curve_path(rod_info, src0)
        if info is None:
            print(f"[warn] {prim.GetPath()}: src0 {src0} is not an imported curve, skipping")
            continue
        indices0 = prim.GetAttribute("physics:indices0").Get() or []
        indices1 = prim.GetAttribute("physics:indices1").Get() or []
        coords1 = prim.GetAttribute("physics:coords1").Get() or []
        if not indices0:
            print(f"[warn] {prim.GetPath()}: no physics:indices0, skipping")
            continue

        info1 = _rod_for_curve_path(rod_info, src1)
        for i, node_idx_raw in enumerate(indices0):
            node_idx = int(node_idx_raw)
            if src1 == "/World":
                # Use a ball joint to world instead of zeroing mass, so the
                # articulation root body keeps its mass and can be simulated normally.
                pinned_bodies = set(info.get("node_to_bodies", {}).get(node_idx, []))
                if not pinned_bodies:
                    body_ids = info["body_indices"]
                    pinned_bodies = {body_ids[min(max(node_idx, 0), len(body_ids) - 1)]}
                c1 = coords1[i] if i < len(coords1) else None
                world_pos = wp.vec3(float(c1[0]), float(c1[1]), float(c1[2])) if c1 is not None else wp.vec3(0.0, 0.0, 0.0)
                for child_body in pinned_bodies:
                    child_local = _node_local_pos(info, builder, child_body, node_idx)
                    builder.add_joint_ball(
                        parent=-1,
                        child=child_body,
                        parent_xform=wp.transform(world_pos, wp.quat_identity()),
                        child_xform=wp.transform(child_local, wp.quat_identity()),
                        label=str(prim.GetPath()).replace("/", "_"),
                    )
                print(f"[attach] {prim.GetName()}: ball-joint {src0}[{node_idx}] to world at {world_pos}")
                continue

            body_ids = info["body_indices"]
            if not body_ids:
                continue
            rod_body = _node_body(info, node_idx)
            if rod_body is None:
                continue
            rod_local_pos = _node_local_pos(info, builder, rod_body, node_idx)

            if info1 is not None:
                # Curve-to-curve attachments were consumed by parse_rods() to
                # build one add_rod_graph() component. Do not add another joint.
                continue

            rigid_id = body_index_map.get(src1)
            if rigid_id is None:
                print(f"[warn] {prim.GetPath()}: external body {src1} not found in builder, skipping")
                continue
            c1 = coords1[i] if i < len(coords1) else wp.vec3(0.0, 0.0, 0.0)
            p1 = wp.vec3(float(c1[0]), float(c1[1]), float(c1[2]))
            builder.add_joint_fixed(
                parent=rigid_id,
                child=rod_body,
                parent_xform=wp.transform(p1, wp.quat_identity()),
                child_xform=wp.transform(rod_local_pos, wp.quat_identity()),
                label=str(prim.GetPath()).replace("/", "_"),
                collision_filter_parent=True,
            )
            print(f"[attach] {prim.GetName()}: {src0}[{node_idx}] -> {src1}")

# ---------------------------------------------------------------------------
# Main simulation class
# ---------------------------------------------------------------------------

class RodSimulation:
    def __init__(self, usd_path: str, viewer, num_steps: int = 200,
                 substeps: int = 10, iterations: int = 5, fps: int = 60):
        self.viewer = viewer
        # Start paused so the user can inspect the scene before simulation begins
        if hasattr(viewer, "_paused"):
            viewer._paused = True
        self.fps = fps
        self.frame_dt = 1.0 / fps
        self.sim_substeps = substeps
        self.sim_iterations = iterations
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        self.num_steps = num_steps

        print(f"\nLoading: {usd_path}")
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            raise RuntimeError(f"Failed to open USD stage: {usd_path}")
        self.stage = stage

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.001
        builder.add_ground_plane(height=-0.1,cfg = newton.ModelBuilder.ShapeConfig(ke=1e5,
                                                                                    kd=1.0, mu=1000.0))

        # First pass: load rigid bodies, joints, and shapes via add_usd.
        # This gives us real bodies with collision + visual shapes.
        # AOUSD curve deformables are parsed below and are skipped by add_usd.
        try:
            usd_result = builder.add_usd(stage, verbose=False)
            body_index_map: dict[str, int] = usd_result["path_body_map"]
            path_shape_map: dict[str, int] = usd_result["path_shape_map"]
            print(f"[add_usd] bodies={len(body_index_map)}  joints={len(usd_result['path_joint_map'])}  shapes={len(path_shape_map)}")
        except Exception as e:
            print(f"[add_usd] failed ({e}), falling back to manual body creation")
            usd_result = None
            body_index_map = {}
            path_shape_map = {}

        # Second pass: parse AOUSD curve deformables.
        rod_info = parse_rods(stage, builder)
        self.rod_info = rod_info
        if not rod_info:
            raise RuntimeError("No PhysicsCurvesDeformableSimAPI BasisCurves found in the USD file.")

        # Wrap any single-cable joints not already wrapped (wrap_in_articulation=False
        # when no ArticulationRoot is found in the hierarchy, e.g. adi assets).
        for info in rod_info.values():
            if info.get("wrap"):
                continue  # already wrapped by add_rod / add_rod_graph
            joint_ids = info.get("joint_indices", [])
            if joint_ids:
                label = info.get("curve_path", "cable").replace("/", "_") + "_articulation"
                builder.add_articulation(joint_ids, label=label)

        # For any attachment target not already in body_index_map, fall back to a
        # static proxy body so simple authored fixtures still load.
        for attachment in stage.Traverse():
            if attachment.GetTypeName() != "PhysicsAttachment":
                continue
            targets = attachment.GetRelationship("physics:src1").GetTargets()
            if not targets:
                continue
            target_path = str(targets[0])
            if target_path == "/World" or target_path in body_index_map:
                continue
            if _rod_for_curve_path(rod_info, target_path) is not None:
                continue
            prim = stage.GetPrimAtPath(target_path)
            if not prim or not prim.IsValid():
                print(f"[warn] attachment target {target_path} not found in stage")
                continue
            xform_api = UsdGeom.Xformable(prim)
            if xform_api:
                mat = xform_api.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = mat.ExtractTranslation()
                pos = wp.vec3(float(t[0]), float(t[1]), float(t[2]))
            else:
                pos = wp.vec3(0.0, 0.0, 0.0)
            body_id = builder.add_body(
                xform=wp.transform(pos, wp.quat_identity()),
                mass=0.0,
                label=target_path.replace("/", "_"),
            )
            body_index_map[target_path] = body_id
            print(f"[body] static proxy for {target_path}: id={body_id}")

        # Parse attachments and local collision filters.
        parse_attachments(stage, builder, rod_info, body_index_map)
        parse_element_collision_filters(stage, builder, rod_info, body_index_map, path_shape_map)

        print("\nFinalizing model...")
        builder.color()
        self.model = builder.finalize()
        print(f"  bodies={self.model.body_count}  joints={self.model.joint_count}  shapes={self.model.shape_count}")

        # Per-body contact buffer: worst case each body touches all others (~n_bodies)
        per_body_contacts = max(64, self.model.body_count * 2)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_body_contact_buffer_size=per_body_contacts,
            rigid_contact_history=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        n_bodies = self.model.body_count
        # rigid_contact_max: total contact pairs; for a coiled rod worst case is O(n^2)
        rigid_contact_max = max(8192, n_bodies * n_bodies)
        pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="latest",
            rigid_contact_max=rigid_contact_max,
        )
        self.contacts = self.model.contacts(collision_pipeline=pipeline)
        self.last_rigid_contact_count = 0
        self.last_soft_contact_count = 0

        self.viewer.set_model(self.model)
        self._set_camera(rod_info)
        self._capture()

    def _set_camera(self, rod_info: dict):
        """Point camera at the bounding box center of all rod nodes."""
        if not hasattr(self.viewer, "set_camera"):
            return
        import math
        all_pts = []
        for info in rod_info.values():
            all_pts.extend(info["positions"])
        if not all_pts:
            return
        xs = [float(p[0]) for p in all_pts]
        ys = [float(p[1]) for p in all_pts]
        zs = [float(p[2]) for p in all_pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        cz = (min(zs) + max(zs)) / 2
        span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 0.1)
        dist = span * 2.5
        # Camera placed at -Y, looking toward +Y (yaw=0°), slightly above target
        cam_pos = wp.vec3(cx, cy - dist, cz + dist * 0.3)
        # Pitch: angle down toward target center
        pitch = -math.degrees(math.atan2(dist * 0.3, dist))
        yaw = 90.0
        self.viewer.set_camera(pos=cam_pos, pitch=pitch, yaw=yaw)
        print(f"  camera: pos=({cam_pos[0]:.3f},{cam_pos[1]:.3f},{cam_pos[2]:.3f})  pitch={pitch:.1f}°  yaw={yaw}°")

    def _capture(self):
        # Contact-count diagnostics read Warp counters back to host each frame,
        # so keep this example runner out of CUDA graph capture.
        self.graph = None

    def _contact_count(self, name: str) -> int:
        value = getattr(self.contacts, name, None)
        if value is None:
            return 0
        try:
            return int(value.numpy()[0])
        except Exception:
            return 0

    def _simulate(self):
        max_rigid_contacts = 0
        max_soft_contacts = 0
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            max_rigid_contacts = max(max_rigid_contacts, self._contact_count("rigid_contact_count"))
            max_soft_contacts = max(max_soft_contacts, self._contact_count("soft_contact_count"))
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.last_rigid_contact_count = max_rigid_contacts
        self.last_soft_contact_count = max_soft_contacts

    def _print_contact_counts(self, frame: int) -> None:
        print(
            f"  frame {frame}  t={self.sim_time:.2f}s  "
            f"contacts rigid={self.last_rigid_contact_count} soft={self.last_soft_contact_count}"
        )

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
        # Sync rod node positions back to BasisCurves visual children
        try:
            body_q_np = self.state_0.body_q.numpy()
            sync_visual_curves(self.stage, self.rod_info, body_q_np)
        except Exception:
            pass  # non-fatal — viewer still shows capsule bodies

    def run(self):
        has_is_running = hasattr(self.viewer, "is_running")
        if has_is_running:
            # Interactive viewer: loop until window closed
            print("\nSimulating (press Space to pause/resume, close window to stop)...")
            frame = 0
            while self.viewer.is_running():
                if not self.viewer.is_paused():
                    self.step()
                    frame += 1
                    self._print_contact_counts(frame)
                self.render()
        else:
            # Headless / file viewer: run fixed number of frames
            print(f"\nSimulating {self.num_steps} frames...")
            for i in range(self.num_steps):
                self.step()
                self.render()
                self._print_contact_counts(i + 1)
        print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate rod/cable USD assets using Newton physics."
    )
    parser.add_argument("usd_file", help="Path to the .usda / .usd file")
    args = parser.parse_args()

    usd_path = str(_resolve_usd_path(args.usd_file))
    viewer = newton.viewer.ViewerGL()

    sim = RodSimulation(
        usd_path=usd_path,
        viewer=viewer,
        num_steps=DEFAULT_STEPS,
        substeps=DEFAULT_SUBSTEPS,
        iterations=DEFAULT_ITERATIONS,
        fps=DEFAULT_FPS,
    )
    sim.run()


if __name__ == "__main__":
    main()
