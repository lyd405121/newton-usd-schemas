#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Generate a NewtonRodAPI graph USD asset from a text string.

Pipeline:
  1. Render each character → grayscale bitmap via FreeType
  2. Threshold + skeletonize (1-pixel-wide medial axis)
  3. Trace skeleton → graph nodes & edges
  4. Prune spurious short branches, merge nearby nodes
  5. Write .usda with one BasisCurves prim per character, all under /World/text_rod

Usage:
    python gen_text_rod.py "NVIDIA"   --out asset/TextNVIDIA.usda
    python gen_text_rod.py "Lightwheel" --out asset/TextLightwheel.usda --font /path/to/font.ttf
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Render text to bitmap
# ---------------------------------------------------------------------------

def render_text_bitmap(
    text: str,
    font_path: str,
    pixel_height: int = 160,
    padding: int = 8,
) -> np.ndarray:
    """Return a 2-D uint8 array (rows, cols) with anti-aliased glyph pixels."""
    import freetype

    face = freetype.Face(font_path)
    face.set_pixel_sizes(0, pixel_height)

    glyphs: list[tuple] = []
    for ch in text:
        face.load_char(ch, freetype.FT_LOAD_RENDER)
        g = face.glyph
        if g.bitmap.width == 0:
            # space / invisible character — store advance only
            glyphs.append((np.zeros((1, 1), np.uint8), 0, 0, g.advance.x >> 6))
            continue
        bm = np.frombuffer(bytes(g.bitmap.buffer), dtype=np.uint8).reshape(
            g.bitmap.rows, g.bitmap.width
        )
        glyphs.append((bm, g.bitmap_left, g.bitmap_top, g.advance.x >> 6))

    total_w = sum(g[3] for g in glyphs) + 2 * padding
    canvas_h = pixel_height + 2 * padding
    canvas = np.zeros((canvas_h, total_w), dtype=np.uint8)

    x = padding
    baseline = pixel_height  # leave room above for ascenders
    for bm, left, top, advance in glyphs:
        r0 = baseline - top + padding
        c0 = x + left
        h, w = bm.shape
        r1, c1 = min(r0 + h, canvas_h), min(c0 + w, total_w)
        rr0, cc0 = max(0, r0), max(0, c0)
        canvas[rr0:r1, cc0:c1] = np.maximum(
            canvas[rr0:r1, cc0:c1],
            bm[rr0 - r0: r1 - r0, cc0 - c0: c1 - c0],
        )
        x += advance

    return canvas


# ---------------------------------------------------------------------------
# Step 2: Threshold + skeletonize
# ---------------------------------------------------------------------------

def skeletonize_bitmap(
    bitmap: np.ndarray,
    threshold: int = 64,
) -> np.ndarray:
    """Return a bool skeleton array (True = skeleton pixel)."""
    from skimage.morphology import skeletonize

    binary = bitmap > threshold
    skel = skeletonize(binary)
    return skel


# ---------------------------------------------------------------------------
# Step 3: Trace skeleton → graph
# ---------------------------------------------------------------------------

def _neighbors8(r: int, c: int, rows: int, cols: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc


def skeleton_to_graph(
    skel: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Trace skeleton pixels into (nodes, edges).

    Nodes are all skeleton pixels.  Edges connect 8-connected neighbours.
    Returns:
        nodes : list of (row, col)
        edges : list of (node_idx_a, node_idx_b) — no duplicates
    """
    rows, cols = skel.shape
    pixel_index: dict[tuple[int, int], int] = {}
    nodes: list[tuple[int, int]] = []

    for r in range(rows):
        for c in range(cols):
            if skel[r, c]:
                pixel_index[(r, c)] = len(nodes)
                nodes.append((r, c))

    edge_set: set[tuple[int, int]] = set()
    for r, c in nodes:
        a = pixel_index[(r, c)]
        for nr, nc in _neighbors8(r, c, rows, cols):
            if (nr, nc) in pixel_index:
                b = pixel_index[(nr, nc)]
                key = (min(a, b), max(a, b))
                edge_set.add(key)

    return nodes, list(edge_set)


# ---------------------------------------------------------------------------
# Step 4: Simplify graph (remove degree-2 chains → single edges)
# ---------------------------------------------------------------------------

def simplify_graph(
    nodes: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    min_branch_pixels: int = 6,
    target_seg_px: int = 5,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Simplify skeleton graph while PRESERVING SHAPE.

    Instead of collapsing degree-2 chains to single edges (which loses curve
    shape), subsample each chain at every ~target_seg_px pixels so that the
    rod still follows the letter contour.

    Steps:
    1. Find key nodes (degree != 2).
    2. For each chain between two key nodes: subsample at target_seg_px spacing.
    3. For isolated all-degree-2 loops: subsample similarly.
    4. Prune stub branches shorter than min_branch_pixels.
    """
    from collections import defaultdict

    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    key_nodes = {i for i in range(len(nodes)) if len(adj[i]) != 2}

    new_nodes: list[tuple[int, int]] = []
    old_to_new: dict[int, int] = {}
    new_edges_set: set[tuple[int, int]] = set()
    consumed: set[int] = set(key_nodes)

    def get_new(old: int) -> int:
        if old not in old_to_new:
            old_to_new[old] = len(new_nodes)
            new_nodes.append(nodes[old])
        return old_to_new[old]

    for k in sorted(key_nodes):
        get_new(k)

    def trace_chain_pixels(start: int, prev: int) -> list[int]:
        """Collect all pixels of a degree-2 chain (not including prev key node)."""
        chain = []
        cur, p = start, prev
        while cur not in key_nodes:
            consumed.add(cur)
            chain.append(cur)
            nexts = [n for n in adj[cur] if n != p]
            if not nexts:
                break
            p, cur = cur, nexts[0]
        # cur is now the end key node (or a dead-end)
        return chain, cur

    def subsample_chain(
        start_key: int,
        chain_pixels: list[int],
        end_key: int,
    ):
        """Add subsampled nodes + edges for one chain."""
        nstart = get_new(start_key)
        nend = get_new(end_key)

        if not chain_pixels:
            # Direct edge between two adjacent key nodes
            if nstart != nend:
                new_edges_set.add((min(nstart, nend), max(nstart, nend)))
            return

        # Compute arc-length along chain
        all_pts = [start_key] + chain_pixels + [end_key]
        arc = [0.0]
        for i in range(1, len(all_pts)):
            ra, ca = nodes[all_pts[i - 1]]
            rb, cb = nodes[all_pts[i]]
            arc.append(arc[-1] + math.hypot(ra - rb, ca - cb))
        total_len = arc[-1]

        if total_len < min_branch_pixels and (
            len(adj[start_key]) == 1 or len(adj[end_key]) == 1
        ):
            return  # prune short stub

        # Choose step count so spacing ≈ target_seg_px
        n_steps = max(1, round(total_len / target_seg_px))
        step_len = total_len / n_steps

        # Sample intermediate points
        sampled_new_ids = [nstart]
        arc_idx = 0
        for s in range(1, n_steps):
            target = s * step_len
            while arc_idx < len(arc) - 1 and arc[arc_idx + 1] < target:
                arc_idx += 1
            # interpolate between all_pts[arc_idx] and all_pts[arc_idx+1]
            p0, p1 = all_pts[arc_idx], all_pts[arc_idx + 1] if arc_idx + 1 < len(all_pts) else all_pts[arc_idx]
            r0, c0 = nodes[p0]
            r1, c1 = nodes[p1]
            seg = arc[arc_idx + 1] - arc[arc_idx] if arc_idx + 1 < len(arc) else 1
            t = (target - arc[arc_idx]) / max(seg, 1e-6)
            ri = round(r0 + t * (r1 - r0))
            ci = round(c0 + t * (c1 - c0))
            new_idx = len(new_nodes)
            new_nodes.append((ri, ci))
            sampled_new_ids.append(new_idx)

        sampled_new_ids.append(nend)

        for i in range(len(sampled_new_ids) - 1):
            a, b = sampled_new_ids[i], sampled_new_ids[i + 1]
            if a != b:
                new_edges_set.add((min(a, b), max(a, b)))

    # Process all chains from key nodes
    processed_pairs: set[tuple[int, int]] = set()
    for k in sorted(key_nodes):
        for nb in list(adj[k]):
            if nb in key_nodes:
                pair = (min(k, nb), max(k, nb))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)
                    subsample_chain(k, [], nb)
            else:
                chain, end = trace_chain_pixels(nb, k)
                if end in key_nodes:
                    pair = (min(k, end), max(k, end))
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        subsample_chain(k, chain, end)

    # Isolated all-degree-2 loops
    for start in range(len(nodes)):
        if start in consumed:
            continue
        loop_raw = []
        cur, prev = start, -1
        visited_loop: set[int] = set()
        while cur not in visited_loop:
            visited_loop.add(cur)
            consumed.add(cur)
            loop_raw.append(cur)
            nexts = [n for n in adj[cur] if n != prev]
            if not nexts:
                break
            prev, cur = cur, nexts[0]
        if len(loop_raw) < 2:
            continue
        is_closed = cur in visited_loop
        # Subsample
        arc = [0.0]
        for i in range(1, len(loop_raw)):
            ra, ca = nodes[loop_raw[i - 1]]
            rb, cb = nodes[loop_raw[i]]
            arc.append(arc[-1] + math.hypot(ra - rb, ca - cb))
        total_len = arc[-1]
        n_steps = max(2, round(total_len / target_seg_px))
        step_len = total_len / n_steps
        sampled = []
        arc_idx = 0
        for s in range(n_steps):
            target = s * step_len
            while arc_idx < len(arc) - 1 and arc[arc_idx + 1] < target:
                arc_idx += 1
            idx = min(arc_idx, len(loop_raw) - 1)
            new_idx = len(new_nodes)
            new_nodes.append(nodes[loop_raw[idx]])
            sampled.append(new_idx)
        for i in range(len(sampled) - 1):
            a, b = sampled[i], sampled[i + 1]
            new_edges_set.add((min(a, b), max(a, b)))
        if is_closed and len(sampled) >= 3:
            new_edges_set.add((min(sampled[0], sampled[-1]), max(sampled[0], sampled[-1])))

    # Re-index
    used = set()
    for a, b in new_edges_set:
        used.add(a); used.add(b)
    remap = {old: i for i, old in enumerate(sorted(used))}
    final_nodes = [new_nodes[o] for o in sorted(used)]
    final_edges = [(remap[a], remap[b]) for a, b in new_edges_set]
    return final_nodes, final_edges


# ---------------------------------------------------------------------------
# Step 5: Merge nearby nodes
# ---------------------------------------------------------------------------

def merge_nearby_nodes(
    nodes: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    min_dist: float = 1.5,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Union-Find merge of nodes within min_dist pixels."""
    n = len(nodes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            ri, ci = nodes[i]
            rj, cj = nodes[j]
            if math.hypot(ri - rj, ci - cj) < min_dist:
                union(i, j)

    # Average positions within each cluster
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    cluster_roots = sorted(clusters.keys())
    new_idx = {root: i for i, root in enumerate(cluster_roots)}
    new_nodes = []
    for root in cluster_roots:
        members = clusters[root]
        r = sum(nodes[m][0] for m in members) / len(members)
        c = sum(nodes[m][1] for m in members) / len(members)
        new_nodes.append((r, c))

    new_edges_set = set()
    for a, b in edges:
        ra, rb = new_idx[find(a)], new_idx[find(b)]
        if ra != rb:
            new_edges_set.add((min(ra, rb), max(ra, rb)))

    return new_nodes, list(new_edges_set)


def keep_largest_component(
    nodes: list[tuple[int, int]],
    edges: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Keep only the largest connected component, discard isolated dots/fragments."""
    if not nodes:
        return nodes, edges
    from collections import defaultdict
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    visited: set[int] = set()
    comps: list[list[int]] = []
    for start in range(len(nodes)):
        if start in visited:
            continue
        comp: list[int] = []
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n); comp.append(n)
            stack.extend(adj[n] - visited)
        comps.append(comp)
    if len(comps) <= 1:
        return nodes, edges
    largest = max(comps, key=len)
    keep = set(largest)
    remap = {old: i for i, old in enumerate(sorted(keep))}
    new_nodes = [nodes[o] for o in sorted(keep)]
    new_edges = [(remap[a], remap[b]) for a, b in edges if a in keep and b in keep]
    removed = len(nodes) - len(new_nodes)
    if removed:
        print(f"    dropped {removed} node(s) in {len(comps)-1} small component(s)")
    return new_nodes, new_edges

def process_char(
    ch: str,
    font_path: str,
    pixel_height: int = 160,
    threshold: int = 64,
    min_branch: int = 8,
    merge_dist: float = 1.5,
    target_seg_px: int = 5,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    """Process one character. Returns (nodes, edges, bitmap_shape).

    Renders the character to its own bitmap, skeletonizes, traces the
    skeleton to a graph, simplifies, and merges nearby nodes.
    Returns the resulting graph and the bitmap dimensions (rows, cols).
    """
    bitmap = render_text_bitmap(ch, font_path, pixel_height=pixel_height)
    skel = skeletonize_bitmap(bitmap, threshold=threshold)
    nodes, edges = skeleton_to_graph(skel)
    if not nodes:
        return [], [], bitmap.shape
    nodes, edges = simplify_graph(nodes, edges, min_branch_pixels=min_branch, target_seg_px=target_seg_px)
    nodes, edges = merge_nearby_nodes(nodes, edges, min_dist=merge_dist)
    nodes, edges = keep_largest_component(nodes, edges)
    return nodes, edges, bitmap.shape


# ---------------------------------------------------------------------------
# Step 7: Write all characters into one USD file
# ---------------------------------------------------------------------------

def _graph_to_strands(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Decompose a rod graph into strands (connected paths) for visual rendering.

    Each edge is covered exactly once. Degree-2 nodes are "passed through";
    strand boundaries occur at junction nodes (degree != 2).

    Handles:
    - Simple linear chains (2 tips, no junctions)
    - Branching graphs (junctions of degree >= 3)
    - Isolated cycles (degree-2 loops, e.g. letter O)
    - Mixed topologies (e.g. letter P: loop + tail)

    Returns a list of node-index sequences, each representing one visual strand.
    """
    from collections import defaultdict

    if not edges:
        return []

    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    degree = {n: len(adj[n]) for n in adj}

    def is_junction(n: int) -> bool:
        return degree.get(n, 0) != 2

    used: set[tuple[int, int]] = set()

    def ekey(u: int, v: int) -> tuple[int, int]:
        return (min(u, v), max(u, v))

    strands: list[list[int]] = []

    def trace(start: int, nxt: int) -> list[int]:
        strand = [start, nxt]
        used.add(ekey(start, nxt))
        cur, prev = nxt, start
        while not is_junction(cur):
            nexts = [n for n in adj[cur] if n != prev]
            if not nexts:
                break
            nb = nexts[0]
            key = ekey(cur, nb)
            if key in used:
                break
            used.add(key)
            strand.append(nb)
            prev, cur = cur, nb
        return strand

    # Strands starting from junction/tip nodes
    for start in sorted(adj.keys()):
        if not is_junction(start):
            continue
        for nb in sorted(adj[start]):
            if ekey(start, nb) not in used:
                strands.append(trace(start, nb))

    # Isolated degree-2 cycles (all nodes degree 2, e.g. letter O)
    for start in sorted(adj.keys()):
        if all(ekey(start, nb) in used for nb in adj[start]):
            continue
        # find an unused edge from start
        unused_nb = next((nb for nb in adj[start] if ekey(start, nb) not in used), None)
        if unused_nb is None:
            continue
        strand = trace(start, unused_nb)
        # close the loop if it returns to start
        if len(strand) > 2 and ekey(strand[-1], start) not in used:
            used.add(ekey(strand[-1], start))
            strand.append(start)
        strands.append(strand)

    return strands


def write_multi_char_usda(
    text: str,
    char_data: list[tuple[list, list, tuple]],  # list of (nodes, edges, bitmap_shape) per char
    out_path: str,
    pixel_height: int = 160,          # must match what was used in process_char
    char_gap: float = 0.01,           # metres between characters
    world_height: float = 0.20,       # metres — target height for all characters
    z_base: float = 0.05,
    rod_width: float = 0.003,
    stretch_stiffness: float = 1000.44,
    stretch_damping: float = 1e-5,
    bend_stiffness: float = 0.824,
    bend_damping: float = 10.0,
    contact_ke: float = 100.0,
    contact_kd: float = 1.0,
    contact_mu: float = 0.5,
):
    """Write all characters as separate Xform+NewtonRodAPI prims into one USD file.

    Each character prim contains:
      - newton:points / newton:radius  (physics geometry)
      - newton:edges / newton:fixedPoints / stiffness attrs
      - def BasisCurves "visual"  (rendering only)
    """
    # Uniform scale: all chars rendered at pixel_height rows
    scale = world_height / pixel_height   # metres per pixel

    total_nodes = sum(len(nd) for nd, _, _ in char_data if nd)
    total_edges = sum(len(ed) for _, ed, _ in char_data if ed)
    n_curves = sum(1 for nd, _, _ in char_data if nd)

    lines = []
    lines.append('#usda 1.0')
    lines.append('(')
    lines.append(f'    doc = """Rod graph asset: text "{text}"')
    lines.append('')
    lines.append('    Generated by gen_text_rod.py from FreeType-rendered bitmaps.')
    lines.append('    Pipeline: render → threshold → skeletonize → simplify → USD.')
    lines.append('')
    lines.append(f'    Characters: {len(text)}  Curves: {n_curves}')
    lines.append(f'    Total nodes: {total_nodes}  Total edges: {total_edges}')
    lines.append(f'    Rod diameter: {rod_width * 1000:.1f}mm"""')
    lines.append('    defaultPrim = "World"')
    lines.append('    metersPerUnit = 1')
    lines.append('    upAxis = "Z"')
    lines.append(')')
    lines.append('')
    lines.append('def Xform "World"')
    lines.append('{')
    lines.append('    def Xform "text_rod"')
    lines.append('    {')

    x_offset = 0.0
    curve_idx = 0

    for char_idx, (ch, (nodes, edges, bitmap_shape)) in enumerate(zip(text, char_data)):
        rows, cols = bitmap_shape
        char_world_width = cols * scale  # natural width of this character in metres

        if not nodes:
            x_offset += char_world_width + char_gap
            continue

        def px_to_world(r, c, _scale=scale, _rows=rows, _x_off=x_offset):
            x = _x_off + c * _scale
            z = z_base + ((_rows - 1) - r) * _scale
            return (round(x, 5), 0.0, round(z, 5))

        world_pts = [px_to_world(r, c) for r, c in nodes]

        # Fixed point: 只固定最顶端的一个节点
        all_rows = [r for r, c in nodes]
        top_row = min(all_rows)
        fixed_idx = next(i for i, (r, c) in enumerate(nodes) if r == top_row)

        n_nodes = len(world_pts)
        pts_str = ", ".join(f"({p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f})" for p in world_pts)
        widths_str = ", ".join([f"{rod_width:.4f}"] * n_nodes)
        edges_str = ", ".join(f"({a}, {b})" for a, b in edges)
        fp_str = str(fixed_idx)
        safe_ch = ch.replace('\\', '\\\\').replace('"', '\\"')

        lines.append(f'        def Xform "curve_{curve_idx}" (  # character \'{safe_ch}\'')
        lines.append('            prepend apiSchemas = ["NewtonRodAPI", "MaterialBindingAPI"]')
        lines.append('        )')
        lines.append('        {')
        lines.append('            rel material:binding = </World/Looks/CablePhysicsMaterial> (')
        lines.append('                bindMaterialAs = "weakerThanDescendants"')
        lines.append('            )')
        lines.append(f'            point3f[] newton:points = [{pts_str}]')
        lines.append(f'            float[] newton:radius = [{rod_width / 2.0:.5f}]')
        lines.append(f'            float newton:contact_ke = {contact_ke}')
        lines.append(f'            float newton:contact_kd = {contact_kd}')
        lines.append('')
        lines.append(f'            int2[] newton:edges = [{edges_str}]')
        lines.append(f'            int[] newton:fixedPoints = [{fp_str}]')
        lines.append('')
        lines.append(f'            float[] newton:stretchStiffness = [{stretch_stiffness}]')
        lines.append(f'            float[] newton:stretchDamping = [{stretch_damping}]')
        lines.append(f'            float[] newton:bendStiffness = [{bend_stiffness}]')
        lines.append(f'            float[] newton:bendDamping = [{bend_damping}]')
        lines.append('')
        lines.append('            bool newton:wrapInArticulation = true')
        lines.append('')

        # Decompose rod graph into visually correct strands
        strands = _graph_to_strands(edges)
        for s_idx, strand in enumerate(strands):
            s_n = len(strand)
            s_pts_str = ', '.join(
                f'({world_pts[i][0]:.5f}, {world_pts[i][1]:.5f}, {world_pts[i][2]:.5f})'
                for i in strand
            )
            s_widths_str = ', '.join([f'{rod_width:.4f}'] * s_n)
            s_indices_str = ', '.join(str(i) for i in strand)
            lines.append(f'            def BasisCurves "visual_{s_idx}" (')
            lines.append('                prepend apiSchemas = ["NewtonRodVisualCurveAPI"]')
            lines.append('            ) {')
            lines.append('                uniform token[] curveType = ["linear"]')
            lines.append(f'                int[] curveVertexCounts = [{s_n}]')
            lines.append(f'                int[] newton:rodPointIndices = [{s_indices_str}]')
            lines.append('                uniform token type = "linear"')
            lines.append('                uniform token wrap = "nonperiodic"')
            lines.append(f'                float[] widths = [{s_widths_str}]')
            lines.append(f'                point3f[] points = [{s_pts_str}]')
            lines.append('            }')
        lines.append('        }')

        x_offset += char_world_width + char_gap
        curve_idx += 1

    lines.append('    }')
    lines.append('')
    lines.append('    def Scope "Looks"')
    lines.append('    {')
    lines.append('        def Material "CablePhysicsMaterial" (')
    lines.append('            prepend apiSchemas = ["PhysicsMaterialAPI", "NewtonMaterialAPI"]')
    lines.append('        )')
    lines.append('        {')
    lines.append(f'            float physics:dynamicFriction = {contact_mu}')
    lines.append(f'            float physics:staticFriction = {contact_mu}')
    lines.append(f'            float newton:contactStiffness = {contact_ke}')
    lines.append(f'            float newton:contactDamping = {contact_kd}')
    lines.append('        }')
    lines.append('    }')
    lines.append('}')
    lines.append('')

    Path(out_path).write_text('\n'.join(lines))
    print(f"Wrote: {out_path}")
    print(f"  characters={len(text)}  curves={n_curves}  total_nodes={total_nodes}  total_edges={total_edges}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def generate(
    text: str,
    out_path: str,
    font: str = DEFAULT_FONT,
    pixel_height: int = 160,
    threshold: int = 64,
    min_branch: int = 8,
    merge_dist: float = 1.5,
    world_height: float = 0.20,
    char_gap: float = 0.01,
    rod_width: float = 0.003,
    debug: bool = False,
):
    """Process each character independently and write a single USD file."""
    print(f"Processing '{text}' character by character at {pixel_height}px...")

    char_data: list[tuple[list, list, tuple]] = []
    for i, ch in enumerate(text):
        print(f"  [{i}] '{ch}' ...", end=" ", flush=True)
        nodes, edges, bitmap_shape = process_char(
            ch,
            font_path=font,
            pixel_height=pixel_height,
            threshold=threshold,
            min_branch=min_branch,
            merge_dist=merge_dist,
        )
        char_data.append((nodes, edges, bitmap_shape))
        if nodes:
            print(f"nodes={len(nodes)}  edges={len(edges)}")
        else:
            print("(skipped — empty glyph)")

        if debug and nodes:
            from PIL import Image
            bitmap = render_text_bitmap(ch, font, pixel_height=pixel_height)
            skel = skeletonize_bitmap(bitmap, threshold=threshold)
            dbg = np.zeros((*bitmap.shape, 3), dtype=np.uint8)
            dbg[:, :, 0] = bitmap
            dbg[skel, 1] = 255
            safe = ch if ch.isalnum() else f"ord{ord(ch)}"
            dbg_path = out_path.replace(".usda", f"_debug_{i}_{safe}.png")
            Image.fromarray(dbg).save(dbg_path)
            print(f"    debug image: {dbg_path}")

    print("Writing USD...")
    write_multi_char_usda(
        text=text,
        char_data=char_data,
        out_path=out_path,
        pixel_height=pixel_height,
        world_height=world_height,
        char_gap=char_gap,
        rod_width=rod_width,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate rod graph USD from text.")
    parser.add_argument("text", help="Text string to render (e.g. NVIDIA)")
    parser.add_argument("--out", default=None, help="Output .usda path")
    parser.add_argument("--font", default=DEFAULT_FONT, help="TrueType font path")
    parser.add_argument("--pixel-height", type=int, default=160)
    parser.add_argument("--threshold", type=int, default=64, help="Binarization threshold (0-255)")
    parser.add_argument("--min-branch", type=int, default=8, help="Prune branches shorter than N pixels")
    parser.add_argument("--merge-dist", type=float, default=1.5, help="Merge nodes within N pixels")
    parser.add_argument("--world-height", type=float, default=0.20, help="Character height in metres (default: 0.20)")
    parser.add_argument("--char-gap", type=float, default=0.01, help="Metres between characters")
    parser.add_argument("--rod-width", type=float, default=0.003, help="Capsule diameter in metres")
    parser.add_argument("--debug", action="store_true", help="Save debug bitmap images per character")
    args = parser.parse_args()

    safe_name = args.text.replace(" ", "_")
    out = args.out or f"asset/Text_{safe_name}.usda"

    generate(
        text=args.text,
        out_path=out,
        font=args.font,
        pixel_height=args.pixel_height,
        threshold=args.threshold,
        min_branch=args.min_branch,
        merge_dist=args.merge_dist,
        world_height=args.world_height,
        char_gap=args.char_gap,
        rod_width=args.rod_width,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()

