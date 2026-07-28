#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate an AOUSD curve-deformable asset with Newton's native USD importer.

Usage:
    python simulate_rod.py <usd_file>

Examples:
    python simulate_rod.py asset/LinearChain.usda
    python simulate_rod.py Graph.usda
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Avoid shadowing the conda-installed Newton package with a checkout under the repo root.
sys.path = [path for path in sys.path if Path(path).resolve() != REPO_ROOT]

import newton  # noqa: E402
import newton.solvers  # noqa: E402
import newton.viewer  # noqa: E402
import warp as wp  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdShade  # noqa: E402

import newton_usd_schemas  # noqa: E402, F401 - registers Newton extension schemas

DEFAULT_STEPS = 200
DEFAULT_SUBSTEPS = 10
DEFAULT_ITERATIONS = 5
DEFAULT_FPS = 120


def _resolve_usd_path(path_arg: str) -> Path:
    """Resolve a USD path from the cwd, script directory, repo, or asset directory."""
    raw_path = Path(path_arg).expanduser()
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [
            Path.cwd() / raw_path,
            SCRIPT_DIR / raw_path,
            REPO_ROOT / raw_path,
            REPO_ROOT / "asset" / raw_path,
        ]
    )

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"USD file not found: {path_arg}\nChecked:\n  - {checked}")


def _bound_physics_material(prim: Usd.Prim) -> Usd.Prim | None:
    material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    material_prim = material.GetPrim()
    return material_prim if material_prim and material_prim.IsValid() else None


def _authored_damping(material_prim: Usd.Prim | None, name: str) -> float | None:
    if material_prim is None:
        return None
    attr = material_prim.GetAttribute(f"newton:{name}Damping")
    if not attr or not attr.HasAuthoredValue():
        return None
    value = float(attr.Get())
    if value == -math.inf:
        return None
    if not math.isfinite(value) or value < 0.0:
        warnings.warn(
            f"{material_prim.GetPath()}: invalid newton:{name}Damping={value}; ignoring it.",
            stacklevel=2,
        )
        return None
    return value


def _apply_cable_damping(stage: Usd.Stage, builder: newton.ModelBuilder, usd_result: dict) -> None:
    """Apply Newton extension damping not yet consumed by ModelBuilder.add_usd()."""
    cable_attrs = usd_result.get("path_cable_attrs", {})
    for path, (_body_ids, joint_ids) in usd_result.get("path_cable_map", {}).items():
        if cable_attrs.get(path, {}).get("damping_imported"):
            continue
        material_prim = _bound_physics_material(stage.GetPrimAtPath(path))
        damping = {name: _authored_damping(material_prim, name) for name in ("stretch", "shear", "bend", "twist")}
        if all(value is None for value in damping.values()):
            continue

        material = cable_attrs.get(path, {}).get("material", {})
        applied: set[str] = set()
        unsupported: set[str] = set()

        for joint in joint_ids:
            linear_dim, angular_dim = builder.joint_dof_dim[joint]
            dof_start = builder.joint_qd_start[joint]

            if (linear_dim, angular_dim) == (1, 1):
                if damping["stretch"] is not None:
                    builder.joint_target_kd[dof_start] = damping["stretch"]
                    applied.add("stretch")
                if damping["bend"] is not None:
                    builder.joint_target_kd[dof_start + 1] = damping["bend"]
                    applied.add("bend")
                unsupported.update(name for name in ("shear", "twist") if damping[name] is not None)
                continue

            if (linear_dim, angular_dim) == (2, 2):
                shear = damping["shear"]
                if shear is None and damping["stretch"] is not None and "shearStiffness" not in material:
                    shear = damping["stretch"]
                twist = damping["twist"]
                if twist is None and damping["bend"] is not None and "twistStiffness" not in material:
                    twist = damping["bend"]

                values = (damping["stretch"], shear, damping["bend"], twist)
                for offset, (name, value) in enumerate(zip(("stretch", "shear", "bend", "twist"), values, strict=True)):
                    if value is not None:
                        builder.joint_target_kd[dof_start + offset] = value
                        applied.add(name)
                continue

            warnings.warn(
                f"{path}: unsupported cable DOF layout ({linear_dim}, {angular_dim}); damping was not applied.",
                stacklevel=2,
            )

        if applied:
            values = ", ".join(f"{name}={damping[name]}" for name in sorted(applied) if damping[name] is not None)
            print(f"[damping] {path}: {values}")
        if unsupported:
            names = ", ".join(sorted(unsupported))
            warnings.warn(
                f"{path}: Newton's two-slot cable layout cannot represent {names} damping; ignoring it.",
                stacklevel=2,
            )


def _curve_world_points(stage: Usd.Stage, cable_paths) -> list[wp.vec3]:
    points: list[wp.vec3] = []
    for path in cable_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.BasisCurves):
            continue
        curves = UsdGeom.BasisCurves(prim)
        world_xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in curves.GetPointsAttr().Get() or []:
            world_point = world_xform.Transform(Gf.Vec3d(*point))
            points.append(wp.vec3(*world_point))
    return points


class RodSimulation:
    def __init__(
        self,
        usd_path: str,
        viewer,
        num_steps: int = DEFAULT_STEPS,
        substeps: int = DEFAULT_SUBSTEPS,
        iterations: int = DEFAULT_ITERATIONS,
        fps: int = DEFAULT_FPS,
    ):
        self.viewer = viewer
        if hasattr(viewer, "_paused"):
            viewer._paused = True
        self.frame_dt = 1.0 / fps
        self.sim_substeps = substeps
        self.sim_iterations = iterations
        self.sim_dt = self.frame_dt / substeps
        self.sim_time = 0.0
        self.num_steps = num_steps

        print(f"\nLoading: {usd_path}")
        self.stage = Usd.Stage.Open(usd_path)
        if not self.stage:
            raise RuntimeError(f"Failed to open USD stage: {usd_path}")

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.001
        builder.add_ground_plane(
            height=-0.1,
            cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0, mu=1000.0),
        )
        usd_result = builder.add_usd(
            self.stage,
            verbose=False,
            schema_resolvers=[newton.usd.SchemaResolverNewton()],
            return_deformable_results=True,
        )
        cable_map = usd_result["path_cable_map"]
        if not cable_map:
            raise RuntimeError("ModelBuilder.add_usd() did not import any cable deformables.")

        _apply_cable_damping(self.stage, builder, usd_result)
        print(f"[add_usd] bodies={len(builder.body_q)} joints={len(builder.joint_type)} shapes={len(builder.shape_body)} cables={len(cable_map)}")

        builder.color()
        self.model = builder.finalize()
        print(f"[model] bodies={self.model.body_count} joints={self.model.joint_count} shapes={self.model.shape_count}")

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_body_contact_buffer_size=max(64, self.model.body_count * 2),
            rigid_contact_history=True,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="latest",
            rigid_contact_max=max(8192, self.model.body_count * self.model.body_count),
        )
        self.contacts = self.model.contacts(collision_pipeline=pipeline)
        self.last_rigid_contact_count = 0
        self.last_soft_contact_count = 0

        self.viewer.set_model(self.model)
        self._set_camera(_curve_world_points(self.stage, cable_map))

    def _set_camera(self, points: list[wp.vec3]) -> None:
        if not points or not hasattr(self.viewer, "set_camera"):
            return
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        zs = [float(point[2]) for point in points]
        center = wp.vec3(
            0.5 * (min(xs) + max(xs)),
            0.5 * (min(ys) + max(ys)),
            0.5 * (min(zs) + max(zs)),
        )
        span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 0.1)
        distance = 2.5 * span
        position = wp.vec3(center[0], center[1] - distance, center[2] + 0.3 * distance)
        pitch = -math.degrees(math.atan2(0.3 * distance, distance))
        self.viewer.set_camera(pos=position, pitch=pitch, yaw=90.0)

    def _contact_count(self, name: str) -> int:
        value = getattr(self.contacts, name, None)
        return int(value.numpy()[0]) if value is not None else 0

    def step(self) -> None:
        max_rigid_contacts = 0
        max_soft_contacts = 0
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            max_rigid_contacts = max(max_rigid_contacts, self._contact_count("rigid_contact_count"))
            max_soft_contacts = max(max_soft_contacts, self._contact_count("soft_contact_count"))
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.last_rigid_contact_count = max_rigid_contacts
        self.last_soft_contact_count = max_soft_contacts
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _print_progress(self, frame: int) -> None:
        print(f"  frame {frame} t={self.sim_time:.2f}s contacts rigid={self.last_rigid_contact_count} soft={self.last_soft_contact_count}")

    def run(self) -> None:
        if hasattr(self.viewer, "is_running"):
            print("\nSimulating (press Space to pause/resume, close window to stop)...")
            frame = 0
            while self.viewer.is_running():
                if not self.viewer.is_paused():
                    self.step()
                    frame += 1
                    self._print_progress(frame)
                self.render()
        else:
            print(f"\nSimulating {self.num_steps} frames...")
            for frame in range(1, self.num_steps + 1):
                self.step()
                self.render()
                self._print_progress(frame)
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a rod/cable USD asset with Newton physics.")
    parser.add_argument("usd_file", help="Path to the .usda or .usd file")
    args = parser.parse_args()

    viewer = newton.viewer.ViewerGL()
    simulation = RodSimulation(str(_resolve_usd_path(args.usd_file)), viewer)
    simulation.run()


if __name__ == "__main__":
    main()
