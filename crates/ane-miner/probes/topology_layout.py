"""Recover exact Zephyr grid routing while retaining the fixture color schedule.

Slots use [2*t, 2*m+1, 2*m+1] channel/y/x order. Routing masks use
[pattern, y, x] order and contain original edge indices, or -1 for padding.
Runtime coefficients never define topology or enter the serialized layout.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Layout:
    node_to_slot: tuple[int, ...]
    slot_to_node: tuple[int, ...]
    color_of_node: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    physical_labels: tuple[int, ...]
    fixture_order: tuple[int, ...]
    m: int
    t: int


@dataclass(frozen=True)
class Pattern:
    destination_channel: int
    source_channel: int
    dx: int
    dy: int


def _slot(label: int, m: int, t: int) -> int:
    side = 2 * m + 1
    r, z = divmod(label, m)
    r, j = divmod(r, 2)
    r, k = divmod(r, t)
    u, w = divmod(r, side)
    x, y = (w, 2 * z + j) if u == 0 else (2 * z + j, w)
    return ((u * t + k) * side + y) * side + x


def build_layout(
    physical_labels: Sequence[int],
    edges: Sequence[tuple[int, int]],
    color_of_node: Sequence[int],
    *,
    m: int,
    t: int,
    fixture_order: Sequence[int] | None = None,
) -> Layout:
    """Map compact IDs through explicit physical labels, without recoloring."""
    if type(m) is not int or m < 1 or type(t) is not int or t not in (2, 4):
        raise ValueError("zephyr shape needs positive m and t in {2, 4}")
    labels = tuple(physical_labels)
    if any(
        type(label) is not int or not 0 <= label < 4 * m * t * (2 * m + 1)
        for label in labels
    ):
        raise ValueError("physical labels are outside the declared zephyr shape")
    slots = tuple(_slot(label, m, t) for label in labels)
    inverse = [-1] * (2 * t * (2 * m + 1) ** 2)
    for node, slot in enumerate(slots):
        if inverse[slot] != -1:
            raise ValueError("physical labels must be unique")
        inverse[slot] = node
    colors = tuple(color_of_node)
    if len(colors) != len(labels):
        raise ValueError("color count differs from node count")
    order = (
        tuple(fixture_order)
        if fixture_order is not None
        else tuple(sorted(range(len(labels)), key=lambda n: colors[n]))
    )
    layout = Layout(slots, tuple(inverse), colors, tuple(edges), labels, order, m, t)
    validate_layout(layout)
    return layout


def validate_layout(layout: Layout) -> None:
    """Reject lost nodes, nonlocal edges, invalid coloring, and schedule drift."""
    m, t = layout.m, layout.t
    if type(m) is not int or m < 1 or type(t) is not int or t not in (2, 4):
        raise ValueError("invalid zephyr shape")
    n = len(layout.physical_labels)
    if not n or len(layout.node_to_slot) != n or len(layout.color_of_node) != n:
        raise ValueError("node mapping and color sizes must agree")
    if any(
        type(value) is not int
        for values in (layout.node_to_slot, layout.slot_to_node, layout.fixture_order)
        for value in values
    ):
        raise ValueError("node mappings and fixture order must contain integers")
    if len(set(layout.physical_labels)) != n or any(
        type(p) is not int or not 0 <= p < 4 * m * t * (2 * m + 1)
        for p in layout.physical_labels
    ):
        raise ValueError("physical labels must be unique and within the zephyr shape")
    expected = tuple(_slot(p, m, t) for p in layout.physical_labels)
    if layout.node_to_slot != expected:
        raise ValueError("node mapping differs from physical labels")
    inverse = [-1] * (2 * t * (2 * m + 1) ** 2)
    for node, slot in enumerate(expected):
        inverse[slot] = node
    if layout.slot_to_node != tuple(inverse):
        raise ValueError("slot mapping must be the inverse node mapping")
    colors = layout.color_of_node
    if any(type(c) is not int or c < 0 for c in colors) or set(colors) != set(
        range(max(colors) + 1)
    ):
        raise ValueError("colors must be contiguous nonnegative integers")
    if sorted(layout.fixture_order) != list(range(n)):
        raise ValueError("fixture order must be a permutation of compact node IDs")
    if [colors[node] for node in layout.fixture_order] != sorted(colors):
        raise ValueError("fixture order must retain sequential color groups")
    seen: set[tuple[int, int]] = set()
    degrees = [0] * n
    for edge in layout.edges:
        if len(edge) != 2 or any(
            type(node) is not int or not 0 <= node < n for node in edge
        ):
            raise ValueError("invalid edge endpoint")
        a, b = edge
        if a >= b or edge in seen:
            raise ValueError("edges must be unique canonical undirected pairs")
        seen.add(edge)
        degrees[a] += 1
        degrees[b] += 1
        if colors[a] == colors[b]:
            raise ValueError("same-color edge violates color independence")
    if max(degrees) > 20:
        raise ValueError("degree exceeds 20")
    patterns, masks = routing_masks(layout)
    if reconstruct_edges(layout, patterns, masks) != layout.edges:
        raise ValueError("routing failed exact edge reconstruction")


def routing_masks(layout: Layout) -> tuple[tuple[Pattern, ...], tuple[int, ...]]:
    """Return fixed shifts and masks in pattern/y/x order; masks index edges."""
    side = 2 * layout.m + 1
    plane_size = side * side
    patterns: list[Pattern] = []
    for channel in range(2 * layout.t):
        u = channel // layout.t
        for step in (-2, -1, 1, 2):
            patterns.append(
                Pattern(channel, channel, step if u else 0, 0 if u else step)
            )
        for source in range((1 - u) * layout.t, (2 - u) * layout.t):
            for dx in (0, 1) if u else (-1, 0):
                for dy in (-1, 0) if u else (0, 1):
                    patterns.append(Pattern(channel, source, dx, dy))
    indices = {pattern: i for i, pattern in enumerate(patterns)}
    mask = [-1] * (len(patterns) * plane_size)
    for edge_index, (a, b) in enumerate(layout.edges):
        for destination, source in ((a, b), (b, a)):
            destination_channel, plane = divmod(
                layout.node_to_slot[destination], plane_size
            )
            y, x = divmod(plane, side)
            source_channel, source_plane = divmod(
                layout.node_to_slot[source], plane_size
            )
            sy, sx = divmod(source_plane, side)
            pattern = Pattern(destination_channel, source_channel, sx - x, sy - y)
            if pattern not in indices:
                raise ValueError(f"edge {(a, b)} has no local routing pattern")
            position = indices[pattern] * plane_size + plane
            if mask[position] != -1:
                raise ValueError("multiple edges occupy one routing position")
            mask[position] = edge_index
    return tuple(patterns), tuple(mask)


def reconstruct_edges(
    layout: Layout, patterns: Sequence[Pattern], masks: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Decode masks independently from edge endpoints and check both directions."""
    side = 2 * layout.m + 1
    plane_size = side * side
    if len(masks) != len(patterns) * plane_size:
        raise ValueError("routing mask length does not match patterns")
    by_edge: dict[int, set[tuple[int, int]]] = {}
    for index, edge in enumerate(masks):
        if type(edge) is not int or edge < -1 or edge >= len(layout.edges):
            raise ValueError("routing mask contains an invalid edge index")
        if edge == -1:
            continue
        pattern_index, plane = divmod(index, plane_size)
        pattern = patterns[pattern_index]
        y, x = divmod(plane, side)
        sx, sy = x + pattern.dx, y + pattern.dy
        if not (0 <= sx < side and 0 <= sy < side):
            raise ValueError("routing mask crosses a grid boundary")
        if not (
            0 <= pattern.destination_channel < 2 * layout.t
            and 0 <= pattern.source_channel < 2 * layout.t
        ):
            raise ValueError("routing mask uses an invalid channel")
        destination = layout.slot_to_node[
            pattern.destination_channel * plane_size + plane
        ]
        source = layout.slot_to_node[
            pattern.source_channel * plane_size + sy * side + sx
        ]
        if min(destination, source) < 0 or destination == source:
            raise ValueError("routing mask reaches padding or a self edge")
        pair = (destination, source)
        directions = by_edge.setdefault(edge, set())
        if pair in directions:
            raise ValueError("routing mask repeats an edge direction")
        directions.add(pair)
    result = []
    for edge in range(len(layout.edges)):
        directions = by_edge.get(edge, set())
        if len(directions) != 2:
            raise ValueError("routing mask must contain both edge directions")
        a, b = min(directions)
        if directions != {(a, b), (b, a)}:
            raise ValueError("routing mask does not pair opposite directions")
        result.append((a, b))
    return tuple(result)


def pack_j(layout: Layout, coefficients: Sequence[float]) -> tuple[float, ...]:
    """Expand runtime J in original edge order to pattern/y/x order."""
    if len(coefficients) != len(layout.edges):
        raise ValueError("J count differs from the explicit edge count")
    _, mask = routing_masks(layout)
    return tuple(0.0 if edge == -1 else float(coefficients[edge]) for edge in mask)


def audit_layout(layout: Layout) -> dict[str, object]:
    validate_layout(layout)
    patterns, mask = routing_masks(layout)
    degrees = Counter(node for edge in layout.edges for node in edge)
    slots = len(layout.slot_to_node)
    side = 2 * layout.m + 1
    return {
        "node_count": len(layout.physical_labels),
        "slot_count": slots,
        "padding_slots": slots - len(layout.physical_labels),
        "edge_count": len(layout.edges),
        "directed_edges": sum(e >= 0 for e in mask),
        "uncovered_edges": 0,
        "max_degree": max(degrees.values(), default=0),
        "color_sizes": [
            layout.color_of_node.count(c) for c in range(max(layout.color_of_node) + 1)
        ],
        "connection_classes": [
            dict(
                asdict(pattern),
                edge_count=sum(
                    e >= 0 for e in mask[i * side * side : (i + 1) * side * side]
                ),
            )
            for i, pattern in enumerate(patterns)
        ],
        "block_occupancy": [
            sum(
                node >= 0
                for node in layout.slot_to_node[i * side * side : (i + 1) * side * side]
            )
            for i in range(2 * layout.t)
        ],
        "logical_state_bytes_fp16_128_reads": slots * 128 * 2,
        "logical_j_bytes_fp16": len(mask) * 2,
        "logical_all_routes_bytes_fp16_128_reads": len(mask) * 128 * 2,
    }


def _routing_json(layout: Layout) -> dict[str, object]:
    patterns, masks = routing_masks(layout)
    return {
        "order": "pattern,y,x",
        "patterns": [asdict(pattern) for pattern in patterns],
        "edge_index_mask": list(masks),
        "color_slot_masks": [
            [
                layout.node_to_slot[n]
                for n in layout.fixture_order
                if layout.color_of_node[n] == c
            ]
            for c in range(max(layout.color_of_node) + 1)
        ],
    }


def save_layout(
    path: Path, layout: Layout, provenance: dict[str, str] | None = None
) -> None:
    validate_layout(layout)
    payload = {
        "schema_version": 1,
        "family": "zephyr",
        "layout": asdict(layout),
        "routing": _routing_json(layout),
        "provenance": provenance or {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_layout(path: Path) -> Layout:
    payload = json.loads(path.read_text())
    if payload["schema_version"] != 1 or payload["family"] != "zephyr":
        raise ValueError("unsupported layout schema or family")
    raw = payload["layout"]
    layout = Layout(
        tuple(raw["node_to_slot"]),
        tuple(raw["slot_to_node"]),
        tuple(raw["color_of_node"]),
        tuple(tuple(edge) for edge in raw["edges"]),
        tuple(raw["physical_labels"]),
        tuple(raw["fixture_order"]),
        raw["m"],
        raw["t"],
    )
    validate_layout(layout)
    if payload["routing"] != _routing_json(layout):
        raise ValueError(
            "serialized routing differs from explicit topology and schedule"
        )
    return layout


def _read_json(path: Path) -> Any:
    content = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    return json.loads(content)


def build_from_sources(
    spec_path: Path,
    edges_path: Path,
    topology_path: Path,
    fixture_meta_paths: Sequence[Path],
) -> Layout:
    """Require metadata and full edge identity before interpreting physical IDs."""
    spec = _read_json(spec_path)
    source = _read_json(topology_path)
    metadata = source["metadata"]
    if metadata["topology_type"] != "zephyr":
        raise ValueError("topology metadata must name zephyr")
    m, t = metadata["topology_shape"]
    if source["properties"]["topology"] != {"type": "zephyr", "shape": [m, t]}:
        raise ValueError("topology metadata and properties disagree")
    labels = tuple(sorted(spec["nodes"]))
    node_index = {node: index for index, node in enumerate(labels)}
    source_edges = {tuple(sorted(edge)) for edge in spec["edges"]}
    if len(node_index) != len(labels) or len(source_edges) != len(spec["edges"]):
        raise ValueError("source graph contains duplicate nodes or edges")
    if (
        len(source["nodes"]) != len(labels)
        or set(source["nodes"]) != set(labels)
        or len(source["edges"]) != len(source_edges)
        or {tuple(sorted(edge)) for edge in source["edges"]} != source_edges
    ):
        raise ValueError("source graph differs from the topology metadata archive")
    if metadata["num_nodes"] != len(labels) or metadata["num_edges"] != len(
        source_edges
    ):
        raise ValueError("topology metadata counts differ from the graph")
    parsed_edges = tuple(
        tuple(map(int, line.split()))
        for line in edges_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if any(len(edge) != 2 for edge in parsed_edges):
        raise ValueError("each explicit edge must contain two endpoints")
    edges = tuple((edge[0], edge[1]) for edge in parsed_edges)
    mapped = {tuple(sorted((node_index[a], node_index[b]))) for a, b in source_edges}
    if len(edges) != len(mapped) or set(edges) != mapped:
        raise ValueError("explicit compact edges do not match the source graph")
    if not fixture_meta_paths:
        raise ValueError("saved fixture metadata is required")
    first = _read_json(fixture_meta_paths[0])
    colors = [-1] * len(labels)
    for color, tile in enumerate(first["tiles"]):
        if tile["length"] != len(tile["nodes"]) or tile["padded"] < tile["length"]:
            raise ValueError("fixture tile length is invalid")
        for node in tile["nodes"]:
            if (
                type(node) is not int
                or not 0 <= node < len(labels)
                or colors[node] != -1
            ):
                raise ValueError("fixture tiles must partition compact node IDs")
            colors[node] = color
    for path in fixture_meta_paths:
        meta = _read_json(path)
        order = [node for tile in meta["tiles"] for node in tile["nodes"]]
        if (
            order != meta["order"]
            or meta["order"] != first["order"]
            or meta["tiles"] != first["tiles"]
        ):
            raise ValueError("fixture order or color schedule differs")
        if (
            meta["nodes"] != len(labels)
            or meta["channels"] < len(labels)
            or meta["channels"] != first["channels"]
        ):
            raise ValueError("fixture node or channel count differs from topology")
        if (meta["groups"], meta["reads"], meta["physical_lanes"]) != (4, 128, 128):
            raise ValueError("fixture must retain four groups and 128 reads")
    return build_layout(labels, edges, colors, m=m, t=t, fixture_order=first["order"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--fixture-meta", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    layout = build_from_sources(args.spec, args.edges, args.topology, args.fixture_meta)
    paths = [args.spec, args.edges, args.topology, *args.fixture_meta]
    provenance = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_layout(args.output / "layout.json", layout, provenance)
    (args.output / "audit.json").write_text(
        json.dumps(audit_layout(layout), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
