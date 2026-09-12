"""Contract tests for #159's roof-family pilot. Synthetic, fast, CPU, no mesh, no h5.

Two seams are pinned independently of any real BuildingWorld mesh or `real.h5` row:

  * `roof_family()` -- the flat/gable/hip/complex classification read off a `fit_program_beam`
    program's own op list and `CutRoof` kinds, against every case its own docstring claims
    precedent for (an all-Layer terrace is NOT flat; a hip mixed with a gable is `gable`, not
    `hip`; two Ramps with no CutRoof is `gable`; a lone Ramp is `complex`, not a fifth bucket).
  * `summarize()` -- the per-population aggregation (fail-stage counts, family fractions,
    dl_ops/dl_planar_fraction medians) on synthetic per-building records, including the two ways a
    real run degenerates: a building that fails before a family is ever assigned, and an empty
    population.

Run: env -u LD_PRELOAD ./venv/bin/python3 scripts/foundations/test_pilot_buildingworld_roof_families.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.foundations import pilot_buildingworld_roof_families as pilot  # noqa: E402
from scripts.foundations.pilot_buildingworld_roof_families import (  # noqa: E402
    PILOT_CITIES, roof_family, summarize,
)


class TestRoofFamily(unittest.TestCase):
    def test_zero_ops_is_flat(self):
        self.assertEqual(roof_family([], set()), "flat")

    def test_a_single_layer_is_flat(self):
        self.assertEqual(roof_family(["Layer"], set()), "flat")

    def test_several_layers_are_complex_not_flat(self):
        """The many-Layer contour-terrace fallback -- exactly the bucket #159 asks whether
        BuildingWorld collapses into. Must never read as `flat`."""
        self.assertEqual(roof_family(["Layer", "Layer", "Layer"], set()), "complex")

    def test_pure_hip_kind_is_hip(self):
        self.assertEqual(roof_family(["CutRoof"], {"hip"}), "hip")

    def test_gable_x_kind_is_gable(self):
        self.assertEqual(roof_family(["CutRoof"], {"gable_x"}), "gable")

    def test_gable_z_kind_is_gable(self):
        self.assertEqual(roof_family(["CutRoof", "Ramp"], {"gable_z"}), "gable")

    def test_hip_mixed_with_gable_kind_reads_as_gable(self):
        """`recover_massing_programs.main()`'s own bridge split: ANY non-hip kind present moves
        the whole program to `gable`, even alongside a hip."""
        self.assertEqual(roof_family(["CutRoof", "CutRoof"], {"hip", "gable_x"}), "gable")

    def test_two_opposing_ramps_with_no_cutroof_is_gable(self):
        self.assertEqual(roof_family(["Ramp", "Ramp"], set()), "gable")

    def test_a_lone_ramp_is_complex_not_a_fifth_bucket(self):
        self.assertEqual(roof_family(["Ramp"], set()), "complex")

    def test_a_lone_layer_ramp_mix_is_complex(self):
        # A single Ramp has no second one to pair into a gable -- the Layer alongside it doesn't
        # change that; falls through to `complex` same as a lone Ramp on its own.
        self.assertEqual(roof_family(["Layer", "Ramp"], set()), "complex")

    def test_two_ramps_on_any_number_of_layers_is_still_gable(self):
        # A `Layer` encodes wall massing (a storey, or a setback -- #4), an axis this classifier
        # does not score; two opposing Ramps is a gable roof on a plain box or a stepped one alike.
        # (Code review raised this as a possible docstring/code mismatch; investigated and
        # confirmed the code is correct -- see roof_family's own docstring.)
        self.assertEqual(roof_family(["Layer", "Ramp", "Ramp"], set()), "gable")
        self.assertEqual(roof_family(["Layer", "Layer", "Ramp", "Ramp"], set()), "gable")


class TestFitAndClassify(unittest.TestCase):
    """`fit_and_classify` calls the real `fit_program_beam` internally, so the fitter itself is
    stubbed out here -- this seam is about what `fit_and_classify` DOES with a program, not
    whether the beam search finds a good one (that is `recover_massing_programs.py`'s own
    coverage)."""

    def _run(self, program):
        with patch.object(pilot, "fit_program_beam", return_value=(program, np.zeros((1, 1)))):
            return pilot.fit_and_classify(np.ones((1, 1), bool), 0, 0, np.zeros((1, 1), int))

    def test_dl_ops_and_planar_fraction_match_roof_description_lengths_own_fields(self):
        program = [dict(op="CutRoof", kind="gable_x"), dict(op="Ramp")]
        rec = self._run(program)
        self.assertEqual(rec["dl_ops"], 2)
        self.assertAlmostEqual(rec["dl_planar_fraction"], 1.0)
        self.assertEqual(rec["family"], "gable")
        self.assertFalse(rec["is_shed"])

    def test_flags_a_lone_ramp_program_as_a_shed(self):
        rec = self._run([dict(op="Ramp")])
        self.assertTrue(rec["is_shed"])
        self.assertEqual(rec["family"], "complex")

    def test_an_already_flat_building_has_zero_ops_and_zero_planar_fraction(self):
        rec = self._run([])
        self.assertEqual(rec["dl_ops"], 0)
        self.assertEqual(rec["dl_planar_fraction"], 0.0)
        self.assertEqual(rec["family"], "flat")


class TestSummarize(unittest.TestCase):
    def test_empty_population_reports_zero_without_a_family_table(self):
        s = summarize([])
        self.assertEqual(s["n_sampled"], 0)
        self.assertEqual(s["n_ok"], 0)
        self.assertNotIn("family_counts", s)

    def test_a_failed_building_is_counted_by_stage_and_excluded_from_family_fractions(self):
        records = [
            dict(ok=True, family="gable", dl_ops=2, dl_planar_fraction=1.0, is_shed=False),
            dict(ok=False, stage="load", error="boom"),
        ]
        s = summarize(records)
        self.assertEqual(s["n_sampled"], 2)
        self.assertEqual(s["n_ok"], 1)
        self.assertEqual(s["n_failed"], 1)
        self.assertEqual(s["fail_stages"], {"load": 1})
        self.assertEqual(s["family_counts"], {"gable": 1})
        self.assertAlmostEqual(s["family_fractions"]["gable"], 1.0)

    def test_family_fractions_sum_to_one_across_a_mixed_population(self):
        records = [
            dict(ok=True, family=fam, dl_ops=k, dl_planar_fraction=0.5, is_shed=False)
            for k, fam in enumerate(["flat", "flat", "gable", "hip", "complex"], start=1)
        ]
        s = summarize(records)
        self.assertAlmostEqual(sum(s["family_fractions"].values()), 1.0)
        self.assertEqual(s["family_counts"]["flat"], 2)
        self.assertEqual(s["dl_ops_median"], 3.0)

    def test_shed_programs_are_counted_separately_from_the_family_table(self):
        records = [
            dict(ok=True, family="complex", dl_ops=1, dl_planar_fraction=1.0, is_shed=True),
            dict(ok=True, family="complex", dl_ops=3, dl_planar_fraction=0.3, is_shed=False),
        ]
        s = summarize(records)
        self.assertEqual(s["n_shed"], 1)
        self.assertEqual(s["family_counts"], {"complex": 2})


class TestPilotCities(unittest.TestCase):
    def test_excludes_exactly_the_anisotropic_and_zero_of_25_watertight_cities(self):
        """#157: Toronto/Mississauga are anisotropic Web Mercator. #159's own cited 0/25-watertight
        spot check: Adelaide, Calgary, Greater Geelong, Perth, Philadelphia, San Francisco,
        Wellington, Yarra. The isotropic-unit-error cities (Boston, Cambridge, New York) must
        stay IN -- a uniform scale error cannot bias which roof family a program fits."""
        self.assertEqual(len(PILOT_CITIES), 9)
        for city in ("Toronto", "Mississauga", "Adelaide", "Calgary", "Greater Geelong", "Perth",
                     "Philadelphia", "San Francisco", "Wellington", "Yarra"):
            self.assertNotIn(city, PILOT_CITIES)
        for city in ("Boston", "Cambridge", "New York", "Berlin", "Cape Town", "Edmonton",
                     "Melbourne", "Montreal", "Tokyo"):
            self.assertIn(city, PILOT_CITIES)


if __name__ == "__main__":
    unittest.main()
