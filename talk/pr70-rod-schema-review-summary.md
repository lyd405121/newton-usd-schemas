# PR #70 Rod Schema Review Discussion Summary

Source: https://github.com/newton-physics/newton-usd-schemas/pull/70

Participants:
- `lyd405121`: author
- `chschuma-disney`: reviewer

## Overall Direction

The reviewer is broadly aligned with adding the Rod schema, but the schema should describe rod physics semantics rather than Newton's current implementation details. In particular, docs and field semantics should avoid hard-coding implementation-specific concepts such as connected capsules, `add_rod`, `add_rod_graph`, or specific engine fallback values when a more abstract schema-level description is possible.

## Simulation Data vs. Visual Curves

The main structural discussion is whether `NewtonRodAPI` should apply directly to `BasisCurves`, or whether the current split should remain:

- `Xform + NewtonRodAPI` stores simulation/topology data.
- Child `BasisCurves + NewtonRodVisualCurveAPI` stores visual/rendering curves.

The reviewer noted that `BasisCurves` is USD's native curve geometry type, so applying rod semantics directly to curves would feel natural for simple rods. The counterpoint is that graph or junction rods are hard to express as a single `BasisCurves` prim, and simulation points do not always match render points. Keeping simulation data on an `Xform` allows lower-resolution simulation, smoother visual curves, skinning, or rigging-driven rendering.

## Agreed or Likely Changes

1. `newton:radius` default wording should be polished.

   The current doc mentions a concrete default value, `0.005`. Reviewer suggested using a more general "engine default" wording, consistent with other schemas. This should be updated.

2. `fixedPoints` and attachments should be unified conceptually.

   The reviewer pointed out that having both `newton:fixedPoints` and `NewtonRodAttachmentAPI` creates two ways to express constraints. A cleaner model is to represent constraints through attachment semantics, with one attachment type corresponding to "attached to world", which is equivalent to fixed points.

3. `youngsModulus` and `poissonRatio` should not use `-1` as an unset sentinel.

   Reviewer suggested using `-inf`, matching other schema conventions. This is especially relevant for Poisson ratio because `-1` is theoretically meaningful as a boundary value, so using it as a sentinel is semantically awkward.

4. `newton:quaternions` should be renamed to `newton:orientations`.

   "Quaternion" describes the representation format, not the physical meaning. The schema field should name the semantic concept, while the doc can state that orientations are represented as quaternions.

5. `newton:closed` needs further decision.

   Reviewer questioned whether `closed` is needed if explicit `newton:edges` can already encode a loop. The current reason for keeping it is compatibility with Newton's `add_rod` API and simpler parser handling for linear rods. A more natural schema design may remove `closed` and require closed loops to be expressed directly through `edges`, but that likely needs a corresponding Newton API cleanup.

## Practical Conclusion

The review outcome is to make the Rod schema more abstract, schema-oriented, and less tied to Newton's current parser/runtime implementation. The strongest design direction is:

- Keep rod physics and visual representation separate unless the project decides to move RodAPI onto `BasisCurves`.
- Move fixed/world constraints toward attachment semantics.
- Replace implementation-specific defaults and sentinels with repo-wide schema conventions.
- Rename representation-oriented fields to semantic names.
- Reconsider whether `closed` belongs in the schema once explicit graph topology is available.
