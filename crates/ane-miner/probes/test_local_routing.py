"""Independent edge-field checks for local routing probes."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import local_routing as local
import numpy as np
import topology_layout as topology


class LocalFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = topology.build_layout(
            (40, 2, 0, 1, 48),
            ((0, 2), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)),
            (0, 1, 2, 0, 0),
            m=2,
            t=2,
            fixture_order=(3, 0, 4, 1, 2),
        )
        self.spins = np.tile(
            np.array([-1, 1, -1, 1, -1], dtype=np.int32)[:, None], (1, 128)
        )
        self.j = np.array([1, -1, 0, 1, -1, 1], dtype=np.int32)
        self.h = np.zeros((5, 128), dtype=np.int32)

    def test_edge_oracle_matches_hand_calculated_fields(self) -> None:
        result = local.edge_fields(self.layout.edges, self.spins, self.j, self.h)
        np.testing.assert_array_equal(result[:, 0], [-1, 0, -4, 1, 0])
        self.h[0] = 1
        result = local.edge_fields(self.layout.edges, self.spins, self.j, self.h)
        np.testing.assert_array_equal(result[:, 0], [0, 0, -4, 1, 0])

    def test_color_fields_match_oracle_with_padding_and_boundaries(self) -> None:
        self.spins[:, 1::2] *= -1
        self.h[:] = np.arange(5)[:, None] % 3 - 1
        job = local.Job(self.spins, self.j, self.h)
        oracle = local.edge_fields(self.layout.edges, self.spins, self.j, self.h)
        for color in range(3):
            expected = np.zeros((128, 128), dtype=np.int32)
            for node, slot in enumerate(self.layout.node_to_slot):
                if self.layout.color_of_node[node] == color:
                    expected[slot] = oracle[node]
            np.testing.assert_array_equal(
                local.routed_fields(self.layout, job, color), expected
            )

    def test_runtime_zero_edge_activation_changes_exact_fields(self) -> None:
        job0 = local.Job(self.spins, self.j, self.h)
        changed = self.j.copy()
        changed[2] = 1
        job1 = local.Job(self.spins, changed, self.h)
        before = local.routed_fields(self.layout, job0, 0)
        after = local.routed_fields(self.layout, job1, 0)
        np.testing.assert_array_equal(after[10] - before[10], np.ones(128))
        self.assertEqual(np.count_nonzero(after != before), 128)

    def test_probe_reuses_program_and_keeps_j_as_an_input(self) -> None:
        job0 = local.Job(self.spins, self.j, self.h)
        job1 = local.Job(self.spins, -self.j, self.h)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = local.prepare_probe(root, self.layout, [job0, job1], color=0)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(manifest["expected"]), 2)
            self.assertEqual(manifest["h_mode"], "zero")
            self.assertEqual(
                [x["name"] for x in manifest["inputs"]], ["a_state", "d_j"]
            )
            self.assertEqual(manifest["eval_repeats"], 10)
            self.assertTrue(manifest["fixed_io"])
            self.assertEqual(proof["cpu_mismatches"], [0, 0])
            program = (root / "program.mil").read_bytes()
            local.prepare_probe(root, self.layout, [job1, job0], color=0)
            self.assertEqual((root / "program.mil").read_bytes(), program)
            self.h[0] = 1
            local.prepare_probe(
                root, self.layout, [local.Job(self.spins, self.j, self.h)], color=0
            )
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["h_mode"], "runtime")
            self.assertEqual(
                [x["name"] for x in manifest["inputs"]], ["a_state", "b_h", "d_j"]
            )

    def test_invalid_color_and_coefficients_are_rejected(self) -> None:
        job = local.Job(self.spins, self.j, self.h)
        with self.assertRaises(ValueError):
            local.routed_fields(self.layout, job, 3)
        with self.assertRaises(ValueError):
            local.routed_fields(
                self.layout, local.Job(self.spins, self.j[:-1], self.h), 0
            )
        with self.assertRaises(ValueError):
            local.routed_fields(
                self.layout, local.Job(self.spins, self.j * 2, self.h), 0
            )


class SweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = topology.build_layout((0, 2), ((0, 1),), (0, 1), m=2, t=2)
        self.job = local.Job(
            np.ones((2, 128), dtype=np.int32),
            np.ones(1, dtype=np.int32),
            np.zeros((2, 128), dtype=np.int32),
        )

    def test_second_color_observes_first_update(self) -> None:
        thresholds = np.zeros((1, 2, 4), dtype=np.int32)
        expected = np.vstack((-np.ones(128), np.ones(128)))
        np.testing.assert_array_equal(
            local.edge_sweeps(self.layout, self.job, thresholds), expected
        )
        np.testing.assert_array_equal(
            local.routed_sweeps(self.layout, self.job, thresholds), expected
        )

    def test_threshold_ties_and_four_groups(self) -> None:
        thresholds = np.full((1, 2, 4), -128, dtype=np.int32)
        thresholds[0, 0] = [-2, -1, 0, -128]
        expected = np.ones((2, 128), dtype=np.int32)
        expected[0, 32:96] = -1
        np.testing.assert_array_equal(
            local.edge_sweeps(self.layout, self.job, thresholds), expected
        )
        np.testing.assert_array_equal(
            local.routed_sweeps(self.layout, self.job, thresholds), expected
        )

    def test_two_chunks_match_combined_run(self) -> None:
        thresholds = np.zeros((2, 2, 4), dtype=np.int32)
        thresholds[1, 0] = [0, -128, 1, -2]
        first = local.routed_sweeps(self.layout, self.job, thresholds[:1])
        second = local.routed_sweeps(
            self.layout, local.Job(first, self.job.j, self.job.h), thresholds[1:]
        )
        np.testing.assert_array_equal(
            second, local.edge_sweeps(self.layout, self.job, thresholds)
        )

    def test_manifest_preloads_threshold_chunks_and_keeps_grid_padding(self) -> None:
        thresholds = np.zeros((2, 2, 4), dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = local.prepare_probe(
                root,
                self.layout,
                [self.job],
                color=None,
                thresholds=[thresholds],
                block_sweeps=1,
            )
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["iterations"], 2)
            self.assertEqual(manifest["eval_repeats"], 1)
            self.assertFalse(manifest["fixed_io"])
            spec = next(
                item for item in manifest["inputs"] if item["name"] == "c_threshold"
            )
            packed = np.fromfile(root / spec["files"][0], np.float16)
            self.assertEqual(packed.size, 2 * spec["elements"])
            expected = np.fromfile(root / manifest["expected"][0], np.float16).reshape(
                128, 128
            )
            np.testing.assert_array_equal(expected[1], np.ones(128))
            self.assertEqual(proof["cpu_mismatches"], [0])
            program = (root / "program.mil").read_bytes()
            local.prepare_probe(
                root,
                self.layout,
                [local.Job(self.job.spins, -self.job.j, self.job.h)],
                color=None,
                thresholds=[thresholds - 1],
                block_sweeps=1,
            )
            self.assertEqual((root / "program.mil").read_bytes(), program)

    def test_partial_chunk_sentinel_leaves_state_unchanged(self) -> None:
        thresholds = np.zeros((3, 2, 4), dtype=np.int32)
        padded = np.concatenate((thresholds, np.full((1, 2, 4), -128, dtype=np.int32)))
        np.testing.assert_array_equal(
            local.edge_sweeps(self.layout, self.job, thresholds),
            local.routed_sweeps(self.layout, self.job, padded),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local.prepare_probe(
                root,
                self.layout,
                [self.job],
                color=None,
                thresholds=[thresholds],
                block_sweeps=2,
            )
            packed = np.fromfile(root / "c_threshold-0.bin", np.float16).reshape(
                2, 128, 32
            )
            np.testing.assert_array_equal(packed[1, :, 4:8], -128)

    def test_nonzero_h_cannot_be_forced_away(self) -> None:
        job = local.Job(self.job.spins, self.job.j, np.ones((2, 128), dtype=np.int32))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            local.prepare_probe(
                Path(directory),
                self.layout,
                [job],
                color=None,
                thresholds=[np.zeros((1, 2, 4), dtype=np.int32)],
                zero_h=True,
            )


@unittest.skipUnless(
    os.environ.get("QUIP_LOCAL_FIXTURES"),
    "set QUIP_LOCAL_FIXTURES for saved job checks",
)
class SavedJobTests(unittest.TestCase):
    def test_two_saved_sweeps_match_original_expected(self) -> None:
        layout = topology.load_layout(
            Path(os.environ["QUIP_TOPOLOGY_ARTIFACTS"]) / "layout.json"
        )
        fixtures = Path(os.environ["QUIP_LOCAL_FIXTURES"])
        for variant in (0, 1):
            fixture = fixtures / f"single-call-real-s2-v{variant}"
            meta = json.loads((fixture / "meta.json").read_text())
            job = local.load_job(layout, fixture)
            thresholds = local.load_thresholds(layout, fixture, 2)
            expected = np.fromfile(fixture / meta["expected_file"], np.float16).reshape(
                meta["channels"], 128
            )
            oracle = local.edge_sweeps(layout, job, thresholds)
            np.testing.assert_array_equal(
                oracle[np.asarray(layout.fixture_order)],
                expected[: len(layout.node_to_slot)],
            )
            np.testing.assert_array_equal(
                local.routed_sweeps(layout, job, thresholds), oracle
            )

    def test_real_saved_fields_and_inactive_edge_replacement(self) -> None:
        artifact = Path(os.environ["QUIP_TOPOLOGY_ARTIFACTS"])
        layout = topology.load_layout(artifact / "layout.json")
        fixtures = Path(os.environ["QUIP_LOCAL_FIXTURES"])
        edge = layout.edges.index((880, 2695))
        for variant in (0, 1):
            job = local.load_job(layout, fixtures / f"single-call-real-s2-v{variant}")
            self.assertEqual(job.j[edge], 0)
            expected = local.edge_fields(layout.edges, job.spins, job.j, job.h)
            for color in (0, 7):
                actual = local.routed_fields(layout, job, color)
                for node, slot in enumerate(layout.node_to_slot):
                    np.testing.assert_array_equal(
                        actual[slot],
                        expected[node]
                        if layout.color_of_node[node] == color
                        else np.zeros(128),
                    )
            changed = job.j.copy()
            changed[edge] = 1
            color = layout.color_of_node[880]
            before = local.routed_fields(layout, job, color)
            after = local.routed_fields(
                layout, local.Job(job.spins, changed, job.h), color
            )
            np.testing.assert_array_equal(
                after[layout.node_to_slot[880]] - before[layout.node_to_slot[880]],
                job.spins[2695],
            )


if __name__ == "__main__":
    unittest.main()
