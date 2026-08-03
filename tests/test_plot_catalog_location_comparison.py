import tempfile
import unittest
from pathlib import Path

import pandas as pd

import plot_catalog_location_comparison as location_plot


class CatalogLocationComparisonTests(unittest.TestCase):
    def test_prepare_catalog_locations_marks_original_and_m_pairs(self):
        catalog = pd.DataFrame(
            {
                "evid": ["CI_40353400", "CI_40353408", "CI_X000000"],
                "skip": [0, 1, 0],
                "label": ["A", "B", "X"],
                "latitude": [33.48, 33.49, 33.481],
                "longitude": [-116.50, -116.51, -116.499],
                "depth": [8.0, 9.0, 10.0],
                "M lat": [33.481, None, None],
                "M lon": [-116.499, None, None],
                "M depth": [8.1, None, None],
            }
        )

        locations = location_plot.prepare_catalog_locations(catalog)

        self.assertEqual(locations["evid_suffix"].tolist(), ["3400", "3408"])
        self.assertEqual(locations["label"].tolist(), ["A", "B"])
        self.assertEqual(locations["has_original"].tolist(), [True, True])
        self.assertEqual(locations["has_m"].tolist(), [True, False])

    def test_prepare_catalog_locations_retains_only_skip_zero_or_one(self):
        catalog = pd.DataFrame(
            {
                "evid": ["CI_40353400", "CI_40353408", "CI_40353416"],
                "skip": [0, 1, 2],
                "label": ["A", "B", "C"],
                "latitude": [33.48, 33.49, 33.50],
                "longitude": [-116.50, -116.51, -116.52],
                "depth": [8.0, 9.0, 10.0],
                "M lat": [33.481, 33.491, 33.501],
                "M lon": [-116.499, -116.509, -116.519],
                "M depth": [8.1, 9.1, 10.1],
            }
        )

        locations = location_plot.prepare_catalog_locations(catalog)

        self.assertEqual(locations["skip"].tolist(), [0, 1])

    def test_plot_catalog_locations_writes_png(self):
        catalog = pd.DataFrame(
            {
                "evid": ["CI_40353400", "CI_40353408"],
                "skip": [0, 1],
                "label": ["A", "B"],
                "latitude": [33.48, 33.49],
                "longitude": [-116.50, -116.51],
                "depth": [8.0, 9.0],
                "M lat": [33.481, 33.491],
                "M lon": [-116.499, -116.509],
                "M depth": [8.1, 9.1],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_path = location_plot.plot_catalog_locations(
                catalog,
                Path(temporary) / "locations.png",
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            m_only_output_path = location_plot.plot_catalog_locations(
                catalog,
                Path(temporary) / "m_locations.png",
                show_catalog_locations=False,
            )
            self.assertTrue(m_only_output_path.is_file())

    def test_parser_accepts_m_only_option(self):
        args = location_plot.build_argument_parser().parse_args(["--m-only"])

        self.assertTrue(args.m_only)

    def test_prepare_catalog_locations_requires_m_columns(self):
        catalog = pd.DataFrame(
            {
                "evid": ["CI_40353400"],
                "skip": [0],
                "label": ["A"],
                "latitude": [33.48],
                "longitude": [-116.50],
            }
        )

        with self.assertRaisesRegex(ValueError, "M lat"):
            location_plot.prepare_catalog_locations(catalog)

    def test_event_annotation_uses_letter_and_optional_evid_suffix(self):
        row = pd.Series({"label": "A", "evid_suffix": "3400"})

        self.assertEqual(location_plot.event_annotation(row, include_evid=False), "A")
        self.assertEqual(location_plot.event_annotation(row, include_evid=True), "A 3400")

    def test_coordinate_formatter_uses_decimal_degrees_without_offset(self):
        self.assertEqual(location_plot.format_decimal_degrees(-116.5, 0), "-116.500")
        self.assertEqual(location_plot.format_decimal_degrees(33.488, 0), "33.488")

    def test_map_title_reflects_whether_catalog_locations_are_shown(self):
        self.assertEqual(location_plot.map_title(True), "Catalog and M event locations")
        self.assertEqual(location_plot.map_title(False), "Relocations")

    def test_plot_requires_label_column_unless_evid_labels_are_requested(self):
        catalog = pd.DataFrame(
            {
                "evid": ["CI_40353400"],
                "skip": [0],
                "latitude": [33.48],
                "longitude": [-116.50],
                "depth": [8.0],
                "M lat": [33.481],
                "M lon": [-116.499],
                "M depth": [8.1],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "label"):
                location_plot.plot_catalog_locations(
                    catalog, Path(temporary) / "locations.png"
                )
            output_path = location_plot.plot_catalog_locations(
                catalog,
                Path(temporary) / "evid_locations.png",
                include_evid_labels=True,
            )
            self.assertTrue(output_path.is_file())
