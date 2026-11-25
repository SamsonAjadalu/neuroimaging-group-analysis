import unittest

import pandas as pd

from group_stats import _complete_cases, parse_analyses
from roi_stats import discover_roi_features


class AnalysisConfigTests(unittest.TestCase):
    def test_parse_supported_analysis_selections(self):
        self.assertEqual(parse_analyses("roi"), ["roi"])
        self.assertEqual(parse_analyses("voxelwise"), ["voxelwise"])
        self.assertEqual(parse_analyses("roi, voxelwise"), ["roi", "voxelwise"])

    def test_reject_invalid_analysis_selections(self):
        for value in ("", "roi,", "roi, roi", "both", "surface"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_analyses(value)

    def test_complete_case_filter_checks_only_selected_variables(self):
        data = pd.DataFrame({
            "subject_id": ["s1", "s2", "s3"],
            "Group": [0, 1, 1],
            "Age": [70.0, 72.0, 69.0],
            "APOE4": [0.0, None, 1.0],
        })
        no_apoe, no_apoe_excluded = _complete_cases(data, ["Group", "Age"], "subject_id")
        apoe, apoe_excluded = _complete_cases(data, ["Group", "Age", "APOE4"], "subject_id")
        self.assertEqual(no_apoe["subject_id"].tolist(), ["s1", "s2", "s3"])
        self.assertEqual(no_apoe_excluded, [])
        self.assertEqual(apoe["subject_id"].tolist(), ["s1", "s3"])
        self.assertEqual(apoe_excluded, ["s2"])


class RoiDiscoveryTests(unittest.TestCase):
    def test_discovers_exact_roi_columns_in_numeric_order(self):
        data = pd.DataFrame(columns=["ROI_10", "ROI_2", "ROI_1", "ROI_3_extra", "other"])
        self.assertEqual(
            discover_roi_features(data),
            [(1, "ROI_1"), (2, "ROI_2"), (10, "ROI_10")],
        )

    def test_rejects_duplicate_numeric_roi_labels(self):
        data = pd.DataFrame(columns=["ROI_1", "ROI_01"])
        with self.assertRaisesRegex(ValueError, "duplicate numeric ROI labels"):
            discover_roi_features(data)


if __name__ == "__main__":
    unittest.main()
