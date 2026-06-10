#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate rod/cable assets defined with NewtonRodAPI from a USD file.

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import newton
import newton.solvers
import newton.viewer
import warp as wp
from pxr import Usd, UsdGeom

import newton_usd_schemas  # noqa: F401 — registers NewtonRodAPI / NewtonRodAttachmentAPI

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

def _scalar_or_uniform(arr, default: float) -> float:
    """Return arr[0] if arr has 1+ elements, else default."""
    if arr and len(arr) >= 1:
        return float(arr[0])
    return default


def _array_or_uniform(arr, n: int, default: float) -> list[float]:
    """Expand arr to length n: length-1 → broadcast, length-n → as-is, empty → default."""
    if not arr or len(arr) == 0:
        return [default] * n
    if len(arr) == 1:
        return [float(arr[0])] * n
    return [float(v) for v in arr]


def parse_rods(stage: Usd.Stage, builder: newton.ModelBuilder) -> dict:
    """Parse all NewtonRodAPI prims from stage into builder.

    Returns a dict mapping prim path → {body_indices, joint_indices, positions,
    segment_half_lengths, edges_list, prim}.
    """
    rod_info: dict = {}

    for prim in stage.Traverse():
        if not prim.HasAPI("NewtonRodAPI"):
            continue

        path = str(prim.GetPath())

        # --- geometry ---
        pts_raw = prim.GetAttribute("newton:points").Get()
        radius_raw = prim.GetAttribute("newton:radius").Get()

        if pts_raw is None:
            print(f"[warn] {path}: no newton:points, skipping")
            continue

        positions = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in pts_raw]
        n_nodes = len(positions)

        # Resolve edges early so we know n_edges for radius expansion
        edges_raw = prim.GetAttribute("newton:edges").Get()
        if edges_raw and len(edges_raw) > 0:
            edges = [(int(e[0]), int(e[1])) for e in edges_raw]
            n_edges = len(edges)
        else:
            edges = None
            n_edges = max(0, n_nodes - 1)

        # Expand newton:radius to per-segment list
        DEFAULT_RADIUS = 0.005
        if not radius_raw or len(radius_raw) == 0:
            radii = [DEFAULT_RADIUS] * n_edges
        elif len(radius_raw) == 1:
            radii = [float(radius_raw[0])] * n_edges
        else:
            radii = [float(r) for r in radius_raw]
            if len(radii) < n_edges:
                radii += [radii[-1]] * (n_edges - len(radii))

        # Representative physics radius (first segment, capped to 45% of seg length)
        rep_radius = radii[0] if radii else DEFAULT_RADIUS
        if n_nodes >= 2:
            seg_len = float(wp.length(positions[1] - positions[0]))
            rep_radius = min(rep_radius, seg_len * 0.45)
        if rep_radius < (radii[0] if radii else DEFAULT_RADIUS):
            print(f"         radius capped: {(radii[0] if radii else DEFAULT_RADIUS)*1000:.2f}mm → {rep_radius*1000:.2f}mm  (seg_len={seg_len*1000:.2f}mm)")

        # --- physics attributes ---
        fp_raw   = prim.GetAttribute("newton:fixedPoints").Get() or []
        ss_raw   = prim.GetAttribute("newton:stretchStiffness").Get()
        sd_raw   = prim.GetAttribute("newton:stretchDamping").Get()
        bs_raw   = prim.GetAttribute("newton:bendStiffness").Get()
        bd_raw   = prim.GetAttribute("newton:bendDamping").Get()
        wrap_art = prim.GetAttribute("newton:wrapInArticulation").Get()
        closed   = prim.GetAttribute("newton:closed").Get() or False

        wrap = bool(wrap_art) if wrap_art is not None else True

        cfg = newton.ModelBuilder.ShapeConfig()

        # --- build rod ---
        if edges is not None:
            # Graph topology via add_rod_graph
            stretch_stiffness = _scalar_or_uniform(ss_raw, 1e5)
            stretch_damping   = _scalar_or_uniform(sd_raw, 0.0)
            bend_stiffness    = _scalar_or_uniform(bs_raw, 0.0)
            bend_damping      = _scalar_or_uniform(bd_raw, 0.0)

            print(f"[rod] {path}: add_rod_graph  nodes={n_nodes}  edges={n_edges}  radius={rep_radius:.4f}")
            body_ids, joint_ids = builder.add_rod_graph(
                node_positions=positions,
                edges=edges,
                radius=rep_radius,
                cfg=cfg,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=stretch_damping,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                wrap_in_articulation=wrap,
                label=path.replace("/", "_"),
            )
        else:
            # Linear chain via add_rod
            n_segments = n_nodes - 1
            stretch_stiffness = _scalar_or_uniform(ss_raw, 1e5)
            stretch_damping   = _scalar_or_uniform(sd_raw, 0.0)
            bend_stiffness    = _scalar_or_uniform(bs_raw, 0.0)
            bend_damping      = _scalar_or_uniform(bd_raw, 0.0)

            print(f"[rod] {path}: add_rod  nodes={n_nodes}  segments={n_segments}  radius={rep_radius:.4f}")
            body_ids, joint_ids = builder.add_rod(
                positions=positions,
                radius=rep_radius,
                cfg=cfg,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=stretch_damping,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                closed=closed,
                wrap_in_articulation=wrap,
                label=path.replace("/", "_"),
            )

        print(f"         bodies={len(body_ids)}  joints={len(joint_ids)}")

        # --- filter all intra-rod self-collisions (adjacent + non-adjacent) ---
        for i, body_a in enumerate(body_ids):
            for body_b in body_ids[i + 1:]:
                for shape_a in builder.body_shapes.get(body_a, []):
                    for shape_b in builder.body_shapes.get(body_b, []):
                        builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))
        print(f"         self-collision pairs filtered: {len(body_ids)*(len(body_ids)-1)//2}")

        # --- fixed points (kinematic) ---
        if fp_raw:
            try:
                from newton._src.sim.enums import BodyFlags
                kinematic_flag = int(BodyFlags.KINEMATIC)
            except Exception:
                kinematic_flag = None

            pinned: set[int] = set()
            if edges is not None:
                # Graph topology
                fixed_nodes = {int(pt) for pt in fp_raw}
                for e_idx, (u, v) in enumerate(edges):
                    if u in fixed_nodes or v in fixed_nodes:
                        if e_idx < len(body_ids):
                            pinned.add(body_ids[e_idx])
            else:
                # Linear chain: body_ids[i] spans from node i to node i+1.
                # A fixed node pins the bodies on both sides of that node.
                for pt_idx in fp_raw:
                    pt_idx = int(pt_idx)
                    if pt_idx < len(body_ids):
                        pinned.add(body_ids[pt_idx])
                    if pt_idx > 0 and (pt_idx - 1) < len(body_ids):
                        pinned.add(body_ids[pt_idx - 1])

            for b in pinned:
                builder.body_mass[b] = 0.0
                builder.body_inv_mass[b] = 0.0
                builder.body_inertia[b] = wp.mat33(0.0)
                builder.body_inv_inertia[b] = wp.mat33(0.0)
                if kinematic_flag is not None and hasattr(builder, "body_flags"):
                    builder.body_flags[b] = kinematic_flag

            print(f"         fixedPoints={list(fp_raw)}  pinned {len(pinned)} bodies")

        # Segment half-lengths and edges_list for visual sync
        if edges is not None:
            edges_list = edges
            half_lengths = [
                float(wp.length(positions[v] - positions[u])) / 2.0
                for u, v in edges_list
            ]
        else:
            edges_list = [(i, i + 1) for i in range(n_nodes - 1)]
            half_lengths = [
                float(wp.length(positions[i + 1] - positions[i])) / 2.0
                for i in range(n_nodes - 1)
            ]

        rod_info[path] = {
            "body_indices": body_ids,
            "joint_indices": joint_ids,
            "positions": positions,
            "half_lengths": half_lengths,
            "edges_list": edges_list,
            "radii": radii,
            "prim": prim,
            "prim_path": path,
        }

    return rod_info


def sync_visual_curves(stage: Usd.Stage, rod_info: dict, body_q_np) -> None:
    """Update BasisCurves children with NewtonRodVisualCurveAPI from current body state.

    For each rod, recovers node world-space positions from body transforms:
        body[e] has position (pos) and orientation (quat), with local +Z pointing
        from node u to node v (edge (u,v)).
        node_u = pos - quat_rotate(quat, (0, 0, half_len))
        node_v = pos + quat_rotate(quat, (0, 0, half_len))

    body_q_np : numpy array of shape (N_bodies, 7) — [px, py, pz, qx, qy, qz, qw]
    """
    import numpy as np
    from pxr import Vt

    def quat_rotate_z(q, half_len):
        """Rotate (0, 0, half_len) by quaternion q = [qx, qy, qz, qw]. Returns np.array(3)."""
        qx, qy, qz, qw = q
        # Optimized rotation of (0,0,z) by unit quaternion
        # R(q) * (0,0,z) = z * (2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx), 1-2*(qx^2+qy^2))
        z = half_len
        rx = 2.0 * (qx * qz + qw * qy) * z
        ry = 2.0 * (qy * qz - qw * qx) * z
        rz = (1.0 - 2.0 * (qx * qx + qy * qy)) * z
        return np.array([rx, ry, rz], dtype=np.float64)

    for path, info in rod_info.items():
        body_ids = info["body_indices"]
        half_lengths = info["half_lengths"]
        edges_list = info["edges_list"]
        radii = info["radii"]
        prim = info["prim"]
        n_nodes = len(info["positions"])

        # Reconstruct current node positions from body transforms
        node_pos = np.zeros((n_nodes, 3), dtype=np.float64)
        node_set = np.zeros(n_nodes, dtype=bool)
        # Per-node radius: use radius of first incident segment
        node_radius = np.full(n_nodes, 0.005, dtype=np.float64)

        for e_idx, (u, v) in enumerate(edges_list):
            if e_idx >= len(body_ids):
                continue
            bid = body_ids[e_idx]
            if bid >= len(body_q_np):
                continue
            bq = body_q_np[bid]
            pos = bq[:3]
            quat = bq[3:]
            hl = half_lengths[e_idx]
            offset = quat_rotate_z(quat, hl)

            if not node_set[u]:
                node_pos[u] = pos - offset
                node_set[u] = True
            if not node_set[v]:
                node_pos[v] = pos + offset
                node_set[v] = True

            r = radii[e_idx] if e_idx < len(radii) else radii[-1]
            node_radius[u] = r
            node_radius[v] = r

        # Write positions and widths to each child BasisCurves with NewtonRodVisualCurveAPI
        for child in prim.GetChildren():
            if not child.HasAPI("NewtonRodVisualCurveAPI"):
                continue
            indices_attr = child.GetAttribute("newton:rodPointIndices")
            if not indices_attr or not indices_attr.HasAuthoredValue():
                continue
            indices = list(indices_attr.Get())
            new_pts = Vt.Vec3fArray([
                (float(node_pos[i, 0]), float(node_pos[i, 1]), float(node_pos[i, 2]))
                for i in indices
            ])
            child.GetAttribute("points").Set(new_pts)
            new_widths = Vt.FloatArray([float(node_radius[i] * 2.0) for i in indices])
            child.GetAttribute("widths").Set(new_widths)


def parse_attachments(stage: Usd.Stage, builder: newton.ModelBuilder,
                      rod_info: dict, body_index_map: dict) -> None:
    """Parse NewtonRodAttachmentAPI children and add fixed joints."""

    for prim in stage.Traverse():
        if not prim.HasAPI("NewtonRodAPI"):
            continue

        rod_path = str(prim.GetPath())
        info = rod_info.get(rod_path)
        if info is None:
            continue

        body_ids = info["body_indices"]

        for child in prim.GetChildren():
            if not child.HasAPI("NewtonRodAttachmentAPI"):
                continue

            node_idx = child.GetAttribute("newton:nodeIndex").Get()
            body_targets = child.GetRelationship("newton:body").GetTargets()
            local_pos1_raw = child.GetAttribute("newton:localPos1").Get()
            local_rot0_raw = child.GetAttribute("newton:localRot0").Get()
            local_rot1_raw = child.GetAttribute("newton:localRot1").Get()

            if node_idx is None:
                print(f"[warn] {child.GetPath()}: missing newton:nodeIndex, skipping")
                continue
            if not body_targets:
                print(f"[warn] {child.GetPath()}: missing newton:body rel, skipping")
                continue

            node_idx = int(node_idx)
            ext_body_path = str(body_targets[0])
            rigid_id = body_index_map.get(ext_body_path)

            if rigid_id is None:
                print(f"[warn] {child.GetPath()}: external body {ext_body_path} not found in builder, skipping")
                continue

            # rod body index: node_idx maps to body_ids[min(node_idx, len-1)]
            rod_body_idx = min(node_idx, len(body_ids) - 1)
            rod_body = body_ids[rod_body_idx]

            # Build transforms — Gf.Quatf stores (w, xi, yj, zk)
            if local_pos1_raw is not None:
                p1 = wp.vec3(float(local_pos1_raw[0]), float(local_pos1_raw[1]), float(local_pos1_raw[2]))
            else:
                p1 = wp.vec3(0.0, 0.0, 0.0)

            def gf_quat_to_wp(q) -> wp.quat:
                im = q.GetImaginary()
                return wp.quat(float(im[0]), float(im[1]), float(im[2]), float(q.GetReal()))

            r0 = gf_quat_to_wp(local_rot0_raw) if local_rot0_raw is not None else wp.quat_identity()
            r1 = gf_quat_to_wp(local_rot1_raw) if local_rot1_raw is not None else wp.quat_identity()

            rod_world_xform = builder.body_q[rod_body]
            rod_node_pos = info["positions"][node_idx]
            rod_local_pos = wp.transform_point(wp.transform_inverse(rod_world_xform), rod_node_pos)

            builder.add_joint_fixed(
                parent=rigid_id,
                child=rod_body,
                parent_xform=wp.transform(p1, r1),
                child_xform=wp.transform(rod_local_pos, r0),
                label=str(child.GetPath()).replace("/", "_"),
                collision_filter_parent=True,
            )
            print(f"[attach] {child.GetName()}: node={node_idx} → body={rod_body}  ext={ext_body_path}")


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
        # NewtonRodAPI prims are unknown to add_usd and are silently skipped.
        try:
            usd_result = builder.add_usd(stage, verbose=False)
            body_index_map: dict[str, int] = usd_result["path_body_map"]
            print(f"[add_usd] bodies={len(body_index_map)}  joints={len(usd_result['path_joint_map'])}  shapes={len(usd_result['path_shape_map'])}")
        except Exception as e:
            print(f"[add_usd] failed ({e}), falling back to manual body creation")
            usd_result = None
            body_index_map = {}

        # Second pass: parse NewtonRodAPI prims
        rod_info = parse_rods(stage, builder)
        self.rod_info = rod_info
        if not rod_info:
            raise RuntimeError("No NewtonRodAPI prims found in the USD file.")

        # For any attachment target not already in body_index_map (e.g. test files
        # whose bodies were loaded by add_usd), fall back to a static proxy body.
        for rod_prim in stage.Traverse():
            if not rod_prim.HasAPI("NewtonRodAPI"):
                continue
            for child in rod_prim.GetChildren():
                if not child.HasAPI("NewtonRodAttachmentAPI"):
                    continue
                targets = child.GetRelationship("newton:body").GetTargets()
                if not targets:
                    continue
                target_path = str(targets[0])
                if target_path in body_index_map:
                    continue  # already loaded by add_usd
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

        # Parse attachments
        parse_attachments(stage, builder, rod_info, body_index_map)

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
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    def _simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

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
                    if frame % 60 == 0:
                        print(f"  frame {frame}  t={self.sim_time:.2f}s")
                self.render()
        else:
            # Headless / file viewer: run fixed number of frames
            print(f"\nSimulating {self.num_steps} frames...")
            for i in range(self.num_steps):
                self.step()
                self.render()
                if (i + 1) % 60 == 0:
                    print(f"  frame {i+1}/{self.num_steps}  t={self.sim_time:.2f}s")
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
