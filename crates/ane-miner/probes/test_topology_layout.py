"""CPU checks for exact topology routing and the saved update schedule."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import topology_layout as topology


class LayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        # Physical labels are deliberately out of order. IDs 1, 2, 3 form a triangle.
        self.labels = (40, 2, 0, 1, 48)
        self.edges = ((0, 2), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4))
        self.colors = (0, 1, 2, 0, 0)
        self.order = (3, 0, 4, 1, 2)

    def layout(self) -> topology.Layout:
        return topology.build_layout(
            self.labels, self.edges, self.colors, m=2, t=2, fixture_order=self.order
        )

    def test_scrambled_labels_map_to_physical_grid(self) -> None:
        layout = self.layout()
        self.assertEqual(layout.node_to_slot, (50, 5, 0, 10, 55))
        self.assertEqual(len(layout.slot_to_node), 100)
        self.assertEqual(layout.slot_to_node[1], -1)
        self.assertEqual(layout.color_of_node, self.colors)
        self.assertEqual(layout.fixture_order, self.order)
        for node, slot in enumerate(layout.node_to_slot):
            self.assertEqual(layout.slot_to_node[slot], node)

    def test_explicit_zero_j_edge_remains_in_both_directions(self) -> None:
        layout = self.layout()
        patterns, masks = topology.routing_masks(layout)
        coefficients = (1, -1, 0, 1, -1, 1)
        self.assertEqual(masks.count(2), 2)
        routed = topology.pack_j(layout, coefficients)
        replacement = topology.pack_j(layout, (1, -1, 1, 1, -1, 1))
        self.assertEqual(sum(a != b for a, b in zip(routed, replacement)), 2)
        self.assertEqual(
            topology.reconstruct_edges(layout, patterns, masks), self.edges
        )

    def test_masks_compute_fields_from_fixed_shifts(self) -> None:
        layout = self.layout()
        spins = (-1, 1, -1, 1, -1)
        coefficients = (1, -1, 0, 1, -1, 1)
        expected = [0] * len(spins)
        for (a, b), j in zip(self.edges, coefficients, strict=True):
            expected[a] += j * spins[b]
            expected[b] += j * spins[a]
        patterns, masks = topology.routing_masks(layout)
        actual = [0] * len(spins)
        for index, edge in enumerate(masks):
            if edge < 0:
                continue
            pattern_index, plane = divmod(index, 25)
            pattern = patterns[pattern_index]
            y, x = divmod(plane, 5)
            destination = layout.slot_to_node[pattern.destination_channel * 25 + plane]
            source = layout.slot_to_node[
                pattern.source_channel * 25 + (y + pattern.dy) * 5 + x + pattern.dx
            ]
            actual[destination] += coefficients[edge] * spins[source]
        self.assertEqual(actual, expected)

    def test_missing_nodes_never_gain_edges(self) -> None:
        layout = self.layout()
        audit = topology.audit_layout(layout)
        self.assertEqual(audit["edge_count"], 6)
        self.assertEqual(audit["directed_edges"], 12)
        self.assertEqual(audit["padding_slots"], 95)
        self.assertEqual(audit["max_degree"], 4)
        self.assertEqual(audit["uncovered_edges"], 0)

    def test_invalid_layouts_fail_before_routing(self) -> None:
        layout = self.layout()
        invalid = (
            replace(layout, node_to_slot=(50, 5, 0, 10, 50)),
            replace(layout, node_to_slot=(50.0, 5, 0, 10, 55)),
            replace(layout, slot_to_node=(-1,) * 100),
            replace(layout, edges=((0, 5),)),
            replace(layout, edges=((0, 0),)),
            replace(layout, edges=((0, 2), (0, 2))),
            replace(layout, color_of_node=(0, 0, 0, 0, 0)),
            replace(layout, fixture_order=(0, 1, 2, 3, 3)),
            replace(layout, physical_labels=(40, 2, 0, 1, 40)),
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                topology.validate_layout(candidate)

    def test_nonlocal_edge_is_rejected(self) -> None:
        # Both labels are legal, but their w positions are too far apart.
        with self.assertRaisesRegex(ValueError, "local"):
            topology.build_layout((0, 32), ((0, 1),), (0, 1), m=2, t=2)

    def test_missing_colors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "color"):
            topology.build_layout((0, 2), ((0, 1),), (0,), m=2, t=2)

    def test_degree_limit_is_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "degree"):
            topology.build_layout(
                tuple(range(22)),
                tuple((0, n) for n in range(1, 22)),
                (0,) + (1,) * 21,
                m=2,
                t=2,
            )

    def test_serialized_layout_checks_routing_and_color_masks(self) -> None:
        layout = self.layout()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            topology.save_layout(path, layout, {"source": "test"})
            self.assertEqual(topology.load_layout(path), layout)
            data = json.loads(path.read_text())
            self.assertEqual(
                data["routing"]["color_slot_masks"], [[10, 50, 55], [5], [0]]
            )
            mask = data["routing"]["edge_index_mask"]
            mask[mask.index(0)] = -1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "routing"):
                topology.load_layout(path)


@unittest.skipUnless(
    os.environ.get("QUIP_TOPOLOGY_ARTIFACTS"),
    "set QUIP_TOPOLOGY_ARTIFACTS for saved fixture checks",
)
class SavedFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(os.environ["QUIP_TOPOLOGY_ARTIFACTS"])
        self.paths = (
            self.root / "advantage2-system1.spec.json",
            self.root / "advantage2-system1.edges",
            self.root / "advantage2_system1.json.gz",
            [self.root / "fixture-v0-meta.json", self.root / "fixture-v1-meta.json"],
        )

    def test_real_graph_keeps_zero_edge_and_saved_color_order(self) -> None:
        layout = topology.build_from_sources(*self.paths)
        meta = json.loads(self.paths[3][0].read_text())
        self.assertEqual(layout.fixture_order, tuple(meta["order"]))
        self.assertEqual(layout.edges[layout.edges.index((880, 2695))], (880, 2695))
        sizes = tuple(layout.color_of_node.count(c) for c in range(8))
        self.assertEqual(sizes, (857, 849, 817, 740, 685, 480, 136, 13))
        patterns, masks = topology.routing_masks(layout)
        self.assertEqual(
            topology.reconstruct_edges(layout, patterns, masks), layout.edges
        )
        self.assertEqual(sum(e >= 0 for e in masks), 83030)
        self.assertEqual(len(patterns), 160)
        audit = topology.audit_layout(layout)
        self.assertEqual((audit["slot_count"], audit["padding_slots"]), (5000, 423))
        self.assertEqual(audit["max_degree"], 20)

    def test_conflicting_fixture_schedule_is_rejected(self) -> None:
        meta = json.loads(self.paths[3][1].read_text())
        meta["order"][0], meta["order"][1] = meta["order"][1], meta["order"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.json"
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "order|schedule"):
                topology.build_from_sources(*self.paths[:3], [self.paths[3][0], path])

    def test_conflicting_topology_metadata_is_rejected(self) -> None:
        data = {"metadata": {"topology_type": "pegasus", "topology_shape": [12, 4]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "zephyr"):
                topology.build_from_sources(
                    self.paths[0], self.paths[1], path, self.paths[3]
                )


if __name__ == "__main__":
    unittest.main()
