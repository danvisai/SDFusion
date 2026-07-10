"""Focused regression checks for the real-element retrieval and realization baseline.

These checks use a temporary element library so they are fast, deterministic, and independent
of the 700 MB production library. Run from the repository root:

  env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python \
      scripts/server/test_element_retrieval_baseline.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "server"))

from scene import element_lib
import element_fit


def _sphere_crop(res=16):
    g = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    return (np.sqrt(x * x + y * y + z * z) - 0.72).astype(np.float16)


class ElementRetrievalBaselineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = Path(self._tmp.name)
        self._old_lib = element_lib.LIB
        element_lib.LIB = self.lib
        self._reset_caches()

    def tearDown(self):
        element_lib.LIB = self._old_lib
        self._reset_caches()
        self._tmp.cleanup()

    @staticmethod
    def _reset_caches():
        element_lib._meta = None
        element_lib._crops = None
        element_lib._cache.clear()
        element_fit._F = None

    def _write_library(self):
        crop = _sphere_crop()
        skeletal = np.full_like(crop, 1.0)
        skeletal.reshape(-1)[:int(skeletal.size * 0.05)] = -1.0

        rows, crops = [], []
        for i in range(24):
            if i < 8:
                rel_y, data = 0.03, skeletal
            elif i < 16:
                rel_y, data = 0.03, crop
            else:
                rel_y, data = 0.30, crop
            rows.append({
                "type": "tower",
                "building": f"RESIDENTIAL_fixture_{i:02d}",
                "cls": "RESIDENTIAL",
                "aspect": [1.0, 1.0],
                "y_frac": 0.8,
                "ext_rel": [rel_y, rel_y, rel_y],
            })
            crops.append(data)
        with open(self.lib / "meta.json", "w") as f:
            json.dump(rows, f)
        np.save(self.lib / "elements_f16.npy", np.stack(crops))

    def _write_solidity_preference_library(self):
        moderate = _sphere_crop()
        g = np.linspace(-1.0, 1.0, moderate.shape[0], dtype=np.float32)
        z, y, x = np.meshgrid(g, g, g, indexing="ij")
        dense = (np.sqrt(x * x + y * y + z * z) - 0.92).astype(np.float16)
        rows, crops = [], []
        for i in range(16):
            rows.append({
                "type": "tower", "building": f"RESIDENTIAL_solidity_{i:02d}",
                "cls": "RESIDENTIAL", "aspect": [1.0, 1.0], "y_frac": 0.8,
                "ext_rel": [0.3, 0.3, 0.3],
            })
            crops.append(moderate if i < 8 else dense)
        with open(self.lib / "meta.json", "w") as f:
            json.dump(rows, f)
        np.save(self.lib / "elements_f16.npy", np.stack(crops))

    def test_retrieval_rebuilds_a_stale_solidity_cache(self):
        self._write_library()
        np.save(self.lib / "solidity.npy", np.asarray([0.1, 0.2], np.float32))

        lib_id, _row = element_fit.retrieve(
            ("tower",), (1.0, 1.0), 0.8,
            building_class="RESIDENTIAL", seed=4, k=8, box_rel_y=0.30,
        )

        self.assertIn(lib_id, range(16, 24))
        solidity = np.load(self.lib / "solidity.npy")
        self.assertEqual(solidity.shape, (24,))

    def test_retrieval_filters_skeletal_crops_and_matches_relative_scale(self):
        self._write_library()

        results = [
            element_fit.retrieve(
                ("tower",), (1.0, 1.0), 0.8,
                building_class="RESIDENTIAL", seed=seed, box_rel_y=0.30,
            )
            for seed in range(32)
        ]
        chosen = [result[0] for result in results]

        self.assertTrue(all(lib_id in range(16, 24) for lib_id in chosen), chosen)
        self.assertTrue(all(result[1]["solidity"] >= element_fit.MIN_SOLIDITY
                            for result in results))
        again = element_fit.retrieve(
            ("tower",), (1.0, 1.0), 0.8,
            building_class="RESIDENTIAL", seed=4, box_rel_y=0.30,
        )[0]
        self.assertEqual(again, chosen[4])

    def test_retrieval_prefers_more_solid_eligible_geometry(self):
        self._write_solidity_preference_library()

        chosen = [
            element_fit.retrieve(
                ("tower",), (1.0, 1.0), 0.8,
                building_class="RESIDENTIAL", seed=seed, k=8, box_rel_y=0.30,
            )[0]
            for seed in range(8)
        ]

        self.assertTrue(all(lib_id in range(8, 16) for lib_id in chosen), chosen)

    def test_element_remains_visible_when_realized_at_output_resolution(self):
        self._write_library()
        from scripts.server.refine import Refiner

        res = 32
        g = np.linspace(-1.0, 1.0, res, dtype=np.float32)
        _z, y, _x = np.meshgrid(g, g, g, indexing="ij")
        base = (y + 0.65).astype(np.float32)  # full-width ground slab; no facade composer
        op = element_fit.element_op(16, [0.0, -0.43, 0.0], [0.18, 0.18, 0.18], smooth=0.0)

        refiner = Refiner.__new__(Refiner)
        refiner.device = "cpu"
        plain = refiner.detail_cube_volume(base, [0.0, 0.0, 0.0], 1.0, res_out=48)
        realized = refiner.detail_cube_volume(
            base, [0.0, 0.0, 0.0], 1.0, res_out=48, detail_edits=[op],
        )

        added = (realized <= 0) & (plain > 0)
        self.assertGreater(int(added.sum()), 100)

    def test_element_recipe_round_trips_and_undo_restores_the_base(self):
        self._write_library()
        import torch
        from scene.sdf_edit import EditableBuilding, EditOp

        def base(points):
            return points[:, 1] + 0.65

        op = element_fit.element_op(16, [0.0, -0.43, 0.0], [0.18, 0.18, 0.18], smooth=0.0)
        original = EditableBuilding(base, [EditOp.from_dict(op)])
        restored = EditableBuilding.from_state(base, original.edit_state())
        points = torch.tensor([[0.0, -0.43, 0.0], [0.8, 0.8, 0.8]])

        self.assertTrue(torch.equal(original.evaluate(points), restored.evaluate(points)))
        undone = restored.undo()
        self.assertEqual(undone.kind, "element")
        self.assertTrue(torch.equal(restored.evaluate(points), base(points)))

    def test_subtractive_element_remains_visible_at_output_resolution(self):
        self._write_library()
        from scripts.server.refine import Refiner

        res = 32
        g = np.linspace(-1.0, 1.0, res, dtype=np.float32)
        _z, y, _x = np.meshgrid(g, g, g, indexing="ij")
        base = (y + 0.65).astype(np.float32)
        op = element_fit.element_op(
            16, [0.0, -0.82, 0.0], [0.15, 0.12, 0.15], smooth=0.0,
        )
        op["mode"] = "subtract"

        refiner = Refiner.__new__(Refiner)
        refiner.device = "cpu"
        plain = refiner.detail_cube_volume(base, [0.0, 0.0, 0.0], 1.0, res_out=48)
        carved = refiner.detail_cube_volume(
            base, [0.0, 0.0, 0.0], 1.0, res_out=48, detail_edits=[op],
        )

        removed = (plain <= 0) & (carved > 0)
        self.assertGreater(int(removed.sum()), 100)

    def test_mixed_element_and_regular_edits_preserve_recipe_order(self):
        self._write_library()
        from scripts.server.refine import Refiner

        res = 32
        g = np.linspace(-1.0, 1.0, res, dtype=np.float32)
        _z, y, _x = np.meshgrid(g, g, g, indexing="ij")
        base = (y + 0.65).astype(np.float32)
        element = element_fit.element_op(
            16, [0.0, -0.43, 0.0], [0.18, 0.18, 0.18], smooth=0.0,
        )
        carve = {
            "kind": "box", "center": [0.0, -0.43, 0.0], "size": [0.12, 0.12, 0.12],
            "mode": "subtract", "smooth": 0.0,
        }

        refiner = Refiner.__new__(Refiner)
        refiner.device = "cpu"
        realized = refiner.detail_cube_volume(
            base, [0.0, 0.0, 0.0], 1.0, res_out=48,
            detail_edits=[element, carve],
        )

        i = lambda value: int(round((value + 1.0) * 0.5 * 47))
        self.assertGreater(float(realized[i(0.0), i(-0.43), i(0.0)]), 0.0)

    def test_invalid_mode_in_deferred_suffix_is_rejected(self):
        self._write_library()
        from scripts.server.refine import Refiner

        g = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
        _z, y, _x = np.meshgrid(g, g, g, indexing="ij")
        base = (y + 0.65).astype(np.float32)
        element = element_fit.element_op(16, [0.0, 0.0, 0.0], [0.2, 0.2, 0.2])
        invalid = {
            "kind": "box", "center": [0.0, 0.0, 0.0], "size": [0.1, 0.1, 0.1],
            "mode": "replace", "smooth": 0.0,
        }
        refiner = Refiner.__new__(Refiner)
        refiner.device = "cpu"

        with self.assertRaisesRegex(ValueError, "mode must be add\\|subtract"):
            refiner.detail_cube_volume(
                base, [0.0, 0.0, 0.0], 1.0, res_out=24,
                detail_edits=[element, invalid],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
