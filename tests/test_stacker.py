import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stacker


class StackerTests(unittest.TestCase):
    def test_resolve_run_directory_adds_implicit_2026_prefix(self):
        with patch.object(stacker, "OUTPUT_ROOT", Path("/example/output")):
            path = stacker.resolve_run_directory("0722_182322_4666")

        self.assertEqual(path, Path("/example/output/20260722_182322_4666"))

    def test_resolve_run_directory_accepts_complete_run_name(self):
        with patch.object(stacker, "OUTPUT_ROOT", Path("/example/output")):
            path = stacker.resolve_run_directory("20260722_182322_4666")

        self.assertEqual(path, Path("/example/output/20260722_182322_4666"))

    def test_resolve_run_directory_rejects_path_or_bad_format(self):
        for value in ("../20260722_182322_4666", "0722", "/tmp/run"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stacker.resolve_run_directory(value)

    def test_normalize_components_preserves_order_and_removes_duplicates(self):
        self.assertEqual(
            stacker.normalize_components(["z", "T", "Z"]),
            ["Z", "T"],
        )

    def test_normalize_components_rejects_unknown_component(self):
        with self.assertRaisesRegex(ValueError, "Unknown component"):
            stacker.normalize_components(["Z", "N"])

    def test_smaller_event_mask_policy_only_masks_lower_magnitudes(self):
        self.assertTrue(stacker.should_mask_other_event(2.0, 1.9, "smaller"))
        self.assertFalse(stacker.should_mask_other_event(2.0, 2.0, "smaller"))
        self.assertFalse(stacker.should_mask_other_event(2.0, 2.1, "smaller"))

    def test_none_event_mask_policy_masks_nothing(self):
        self.assertFalse(stacker.should_mask_other_event(2.0, 1.0, "none"))

    def test_has_zero_skip_accepts_only_numeric_zero(self):
        for value in (0, 0.0, "0", "0.0"):
            with self.subTest(value=value):
                self.assertTrue(stacker.has_zero_skip(value))
        for value in (None, "", " ", 1, 2, "1", "other"):
            with self.subTest(value=value):
                self.assertFalse(stacker.has_zero_skip(value))

    def test_argument_free_run_uses_editor_defaults(self):
        args = stacker.build_argument_parser().parse_args([])

        self.assertEqual(args.run, stacker.DEFAULT_RUN)
        self.assertEqual(args.components, stacker.DEFAULT_COMPONENTS)
        self.assertEqual(
            args.mask_other_events,
            stacker.DEFAULT_MASK_OTHER_EVENTS,
        )
        self.assertEqual(args.overlay_only, stacker.DEFAULT_OVERLAY_ONLY)
        self.assertEqual(args.show, stacker.DEFAULT_SHOW_PLOTS)
        self.assertEqual(args.save, stacker.DEFAULT_SAVE_PLOTS)
        self.assertEqual(args.amplitude_limits, stacker.DEFAULT_AMPLITUDE_LIMITS)

    def test_command_line_can_override_editor_defaults(self):
        args = stacker.build_argument_parser().parse_args(
            [
                "--run",
                "0722_182322_4666",
                "--components",
                "Z",
                "T",
                "--mask-other-events",
                "all",
                "--all-plots",
                "--no-show",
                "--save",
                "--amplitude-limits",
                "-0.2",
                "0.2",
            ]
        )

        self.assertEqual(args.run, "0722_182322_4666")
        self.assertEqual(args.components, ["Z", "T"])
        self.assertEqual(args.mask_other_events, "all")
        self.assertFalse(args.overlay_only)
        self.assertFalse(args.show)
        self.assertTrue(args.save)
        self.assertEqual(args.amplitude_limits, [-0.2, 0.2])

    def test_load_run_parameters_rejects_inconsistent_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            first_dir = run_dir / "E1"
            second_dir = run_dir / "E2"
            first_dir.mkdir()
            second_dir.mkdir()
            (first_dir / "rp_1.json").write_text(
                '{"min_freq": 1, "max_freq": 4}',
                encoding="utf-8",
            )
            (second_dir / "rp_2.json").write_text(
                '{"min_freq": 3, "max_freq": 10}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "snapshots disagree"):
                stacker.load_run_parameters(run_dir)


if __name__ == "__main__":
    unittest.main()
