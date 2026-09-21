"""Checks for paired study receipt validation and block-level inference."""

import unittest

from analyze_coloring_router import compare, pool, validate


def receipt(shift=0, wall=2.0, color="greedy"):
    jobs = []
    for block in [1, 2, 3]:
        for seed in [10, 11]:
            jobs.append(
                {
                    "kind": "job",
                    "block": block,
                    "replicate": seed,
                    "reads": 64,
                    "sweeps": 14336,
                    "engine": "metal",
                    "color": color,
                    "best_milli": -100 + shift,
                    "target_milli": -100,
                    "winner_milli": -110,
                    "valid_reads": int(shift < 0),
                    "dispatch_s": 0.0,
                    "completed_s": wall,
                    "device_us": 100,
                    "rescore_ok": True,
                    "warmup": False,
                }
            )
    summary = {
        "kind": "summary",
        "engine": "metal",
        "color": color,
        "reads": 64,
        "sweeps": 14336,
        "window_wall_s": wall,
        "jobs": 6,
        "completed": 6,
        "valid_jobs": 6 if shift < 0 else 0,
        "rescore_ok": True,
    }
    return jobs + [summary]


class StudyAnalysisTests(unittest.TestCase):
    def test_pools_windows_without_pair_collisions_or_clock_mixups(self):
        first = receipt(-1, wall=2)
        second = receipt(-1, wall=4)
        for row in second[:-1]:
            row["replicate"] += 100
            row["dispatch_s"] += 1000
            row["completed_s"] += 1000
        merged = pool([first, second])
        indexed, summary = validate(merged)
        self.assertEqual(len(indexed), 12)
        self.assertEqual(summary["window_wall_s"], 6)
        from analyze_coloring_router import arm_summary

        arm = arm_summary(indexed, summary)
        self.assertEqual(arm["valid_jobs_per_s"], 2)
        self.assertEqual(arm["first_valid_median_s_successful_cohorts"], 3)
        self.assertEqual(len(arm["windows"]), 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate(pool([first, first]))
        with self.assertRaisesRegex(ValueError, "different arms"):
            pool([first, receipt(color="four")])

    def test_rejects_mislabeled_or_identical_arms(self):
        rows = receipt()
        rows[0]["color"] = "four"
        with self.assertRaisesRegex(ValueError, "color"):
            validate(rows)
        with self.assertRaisesRegex(ValueError, "identical"):
            compare(receipt(), receipt(), draws=100)

    def test_bootstrap_retains_block_clusters(self):
        before = receipt(-100)
        after = receipt(-100, color="four")
        for row in after[:-1]:
            row["best_milli"] += (row["block"] - 2) * 10
        result = compare(before, after, draws=2000)
        self.assertEqual(result["energy_delta_milli"]["ci95"], [-10, 10])

    def test_rejects_unscored_and_incomplete_receipts(self):
        for mutation in ["score", "summary", "duplicate", "count"]:
            rows = receipt()
            if mutation == "score":
                rows[0]["rescore_ok"] = False
            elif mutation == "summary":
                rows.pop()
            elif mutation == "duplicate":
                rows[1] = rows[0].copy()
            else:
                rows[-1]["completed"] = 5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(rows)

    def test_strict_target_and_censoring(self):
        result = compare(receipt(), receipt(-1, 1.0, "four"), draws=100)
        self.assertEqual(result["control"]["valid_jobs"], 0)
        self.assertIsNone(result["control"]["seconds_per_valid_job"])
        self.assertEqual(result["candidate"]["valid_jobs"], 6)
        self.assertEqual(result["energy_delta_milli"]["mean"], -1)
        self.assertEqual(result["energy_delta_milli"]["ci95"], [-1, -1])
        self.assertEqual(result["valid_probability_delta"]["ci95"], [1, 1])
        self.assertEqual(result["control"]["blocks_without_valid_result"], 3)

    def test_rejects_mismatched_pairs(self):
        other = receipt()
        other[0]["replicate"] = 12
        with self.assertRaisesRegex(ValueError, "pairs"):
            compare(receipt(), other, draws=100)

    def test_read_count_change_must_be_explicit(self):
        other = receipt()
        for row in other:
            row["reads"] = 128
        with self.assertRaisesRegex(ValueError, "read"):
            compare(receipt(), other, draws=100)
        result = compare(receipt(), other, draws=100, allow_read_change=True)
        self.assertEqual(result["pairs"], 6)


if __name__ == "__main__":
    unittest.main()
