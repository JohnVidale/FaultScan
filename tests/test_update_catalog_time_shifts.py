import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.update_catalog_time_shifts import (
    apply_updates_to_workbook,
    compute_catalog_updates,
)


class UpdateCatalogTimeShiftsTests(unittest.TestCase):
    def shifts(self, measured=True):
        return pd.DataFrame(
            {
                "event_id": ["E1"],
                "shift_left_to_align_waveform_seconds": [0.02],
                "xcorr_residual_measured": [measured],
                "catalog_time_shift_seconds": [0.10],
                "catalog_time_shift_column": ["time shift"],
            }
        )

    def test_combines_catalog_baseline_with_measured_residual(self):
        catalog = pd.DataFrame({"evid": ["E1"], "time shift": [0.10]})
        updates, source = compute_catalog_updates(catalog, self.shifts(), "time shift")

        self.assertEqual(source, "time shift")
        self.assertEqual(updates[0]["action"], "update")
        self.assertAlmostEqual(updates[0]["proposed_seconds"], 0.12)

    def test_repeated_update_is_a_no_op(self):
        catalog = pd.DataFrame({"evid": ["E1"], "time shift": [0.12]})
        updates, _source = compute_catalog_updates(catalog, self.shifts(), "time shift")

        self.assertEqual(updates[0]["action"], "unchanged")

    def test_rejects_unmeasured_alignment_workbook(self):
        catalog = pd.DataFrame({"evid": ["E1"], "time shift": [0.10]})
        with self.assertRaisesRegex(ValueError, "without measured"):
            compute_catalog_updates(catalog, self.shifts(measured=False), "time shift")

    def test_rejects_stale_catalog_baseline(self):
        catalog = pd.DataFrame({"evid": ["E1"], "time shift": [0.30]})
        with self.assertRaisesRegex(ValueError, "regenerate alignment"):
            compute_catalog_updates(catalog, self.shifts(), "time shift")

    def test_applies_validated_update_to_workbook(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.xlsx"
            pd.DataFrame({"evid": ["E1"], "time shift": [0.10]}).to_excel(
                catalog_path, index=False
            )
            updates, _source = compute_catalog_updates(
                pd.read_excel(catalog_path), self.shifts(), "time shift"
            )

            count = apply_updates_to_workbook(catalog_path, "time shift", updates)
            result = pd.read_excel(catalog_path)

        self.assertEqual(count, 1)
        self.assertAlmostEqual(result.loc[0, "time shift"], 0.12)


if __name__ == "__main__":
    unittest.main()
