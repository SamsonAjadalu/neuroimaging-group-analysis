from configparser import ConfigParser
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import voxelwise_stats
from group_stats import model_predictors, run_analysis
from prepare_data import prepare_data
from run_analysis import load_config


class SharedModelTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        rng = np.random.default_rng(43)
        rows = list(product(("a", "b", "c", "control"), ("no", "yes"), range(4)))
        self.data = pd.DataFrame(rows, columns=["group", "exposure", "rep"])
        self.data["id"] = [f"s{i:02d}" for i in range(len(rows))]
        self.data["age"] = rng.normal(size=len(rows))
        self.data["score"] = rng.normal(size=len(rows))
        self.data["interaction"] = [
            group if group != "control" and exposure == "yes" else "none"
            for group, exposure, _ in rows
        ]
        self.group = np.column_stack([self.data.group == level for level in ("a", "b", "c")])
        self.exposure = (self.data.exposure == "yes").to_numpy()
        self.interaction = self.group * self.exposure[:, None]
        self.ones = np.ones(len(rows))
        self.y = rng.normal(size=(len(rows), 6))
        self.y += np.column_stack((self.group, self.interaction)) @ np.diag([2, -2, 1, -2, 2, 1])
        self.y += self.data.age.to_numpy()[:, None] * 0.4 + self.exposure[:, None] * 0.6
        self.roi_y = self.y.reshape(len(rows), 3, 2).mean(axis=2)
        for i, values in enumerate(self.roi_y.T, 1):
            self.data[f"ROI_{i}"] = values
        self.settings = {"p_threshold": ".05", "multiple_comparisons": "fdr_bh", "q_threshold": ".05"}

    def config(self, model, references, modules="roi"):
        self.data.to_csv(self.root / "source.csv", index=False)
        parser = ConfigParser()
        parser.optionxform = str
        parser.read_dict({
            "paths": {"source": "source.csv", "atlas": "atlas.nii.gz", "mask": "mask.nii.gz", "output": "out"},
            "imaging": {"subject_id_column": "id", "image_pattern": "images/{subject}.nii.gz"},
            "analysis": {"analysis": modules},
            "model": model,
            "references": references,
            "roi_statistics": self.settings,
            "voxelwise_statistics": self.settings,
        })
        path = self.root / "model.txt"
        with path.open("w") as stream:
            parser.write(stream)
        return load_config(path)

    def roi_run(self, model, refs):
        config = self.config(model, refs)
        return run_analysis(config, self.root / "source.csv")

    def check_roi(self, result, x, indices, predictor):
        fits = [sm.OLS(y, x).fit() for y in self.roi_y.T]
        table = result["results"]
        if len(indices) > 1:
            tests = [fit.f_test(np.eye(x.shape[1])[indices]) for fit in fits]
            np.testing.assert_allclose(table["F"], [test.fvalue for test in tests])
            np.testing.assert_allclose(table["p"], [test.pvalue for test in tests])
            np.testing.assert_array_equal(table["df_num"], len(indices))
            np.testing.assert_array_equal(table["df_den"], len(x) - x.shape[1])
            coefficients = list(result["level_results"].groupby("design_column", sort=False))
        else:
            coefficients = [(predictor, table.rename(columns={f"beta_{predictor}": "beta"}))]
        np.testing.assert_allclose(table["q"], multipletests(table["p"], method="fdr_bh")[1])
        self.assertEqual(len(coefficients), len(indices))
        for (_, family), idx in zip(coefficients, indices):
            expected = [[fit.params[idx], fit.bse[idx], fit.tvalues[idx], fit.pvalues[idx],
                         *fit.conf_int()[idx]] for fit in fits]
            np.testing.assert_allclose(family[["beta", "SE", "t", "p", "CI_low", "CI_high"]], expected)
            q = multipletests(np.array(expected)[:, 3], method="fdr_bh")[1]
            np.testing.assert_allclose(family["q"], q)
            np.testing.assert_array_equal(family["FDR_survives_q05"], q < .05)

    def test_existing_singular_continuous_and_intercept(self):
        for name, x, idx in (
            ("age", np.column_stack((self.ones, self.data.age)), 1),
            ("intercept", self.ones[:, None], 0),
        ):
            with self.subTest(name=name):
                result = self.roi_run({"primary_predictor": name}, {})
                self.check_roi(result["roi"], x, [idx], name)
                self.assertTrue((self.root / "out/roi/roi_results.csv").is_file())

    def test_existing_singular_categorical(self):
        result = self.roi_run({"primary_predictor": "group"}, {"group": "control"})
        self.check_roi(result["roi"], np.column_stack((self.ones, self.group)), [1, 2, 3], "group")
        self.assertTrue((self.root / "out/roi/primary_level_results.csv").is_file())

    def test_multiple_continuous_predictors(self):
        result = self.roi_run({"primary_predictors": "age, score"}, {})
        x = np.column_stack((self.ones, self.data.age, self.data.score))
        for i, name in enumerate(("age", "score"), 1):
            self.check_roi(result["roi"][name], x, [i], name)

    def test_mixed_primary_predictors(self):
        result = self.roi_run({"primary_predictors": "age, group", "covariates": "score"},
                              {"group": "control"})
        x = np.column_stack((self.ones, self.group, self.data.age, self.data.score))
        self.check_roi(result["roi"]["age"], x, [4], "age")
        self.check_roi(result["roi"]["group"], x, [1, 2, 3], "group")
        self.assertFalse((self.root / "out/roi/score").exists())

    def test_multiple_categorical_interaction_roi_and_voxelwise(self):
        config = self.config(
            {"primary_predictors": "group, exposure, interaction", "covariates": "age"},
            {"group": "control", "exposure": "no", "interaction": "none"},
            "roi, voxelwise",
        )
        shape = (2, 2, 2)
        affine = np.diag([2., 2., 2., 1.])
        nib.save(nib.Nifti1Image(np.array([1, 1, 2, 2, 3, 3, 0, 0], dtype=np.int16).reshape(shape), affine),
                 config["paths"]["atlas"])
        nib.save(nib.Nifti1Image(np.array([1]*6 + [0, 0], dtype=np.uint8).reshape(shape), affine),
                 config["paths"]["mask"])
        (self.root / "images").mkdir()
        for subject, y in zip(self.data.id, self.y):
            nib.save(nib.Nifti1Image(np.r_[y, 0, 0].reshape(shape), affine),
                     self.root / f"images/{subject}.nii.gz")
        # Preparation creates the ROI columns itself for this end-to-end example.
        self.data.drop(columns=["ROI_1", "ROI_2", "ROI_3"]).to_csv(config["paths"]["source"], index=False)
        prepared = self.root / "out/prepared_data.csv"
        prepare_data(config, prepared)
        # Image preparation reads float32 voxel values before averaging in float64.
        self.roi_y = self.y.astype(np.float32).reshape(len(self.y), 3, 2).mean(axis=2, dtype=np.float64)
        original_fit = sm.OLS.fit
        with (patch.object(sm.OLS, "fit", autospec=True, side_effect=original_fit) as roi_fit,
              patch.object(voxelwise_stats, "run_glm", wraps=voxelwise_stats.run_glm) as voxel_fit):
            result = run_analysis(config, prepared)
        self.assertEqual(roi_fit.call_count, 3)
        self.assertEqual(voxel_fit.call_count, 1)

        # Construct the reference design directly from group and exposure indicators.
        x = np.column_stack((self.ones, self.group, self.exposure, self.interaction, self.data.age))
        fits = [sm.OLS(y, x).fit() for y in self.y.T]
        for name, indices in {"group": [1, 2, 3], "exposure": [4], "interaction": [5, 6, 7]}.items():
            self.check_roi(result["roi"][name], x, indices, name)
            folder = self.root / "out/voxelwise" / name
            if len(indices) > 1:
                tests = [fit.f_test(np.eye(x.shape[1])[indices]) for fit in fits]
                self.check_maps(folder, name, {"F": [test.fvalue for test in tests],
                                              "p": [test.pvalue for test in tests]}, affine)
                index = pd.read_csv(folder / "coefficient_index.csv")
                self.assertEqual(index.design_column.tolist(), list(result["voxelwise"][name]["level_results"]))
                targets = [folder / folder_name for folder_name in index.directory]
            else:
                targets = [folder]
            for destination, idx in zip(targets, indices):
                expected = np.array([[fit.params[idx], fit.bse[idx], fit.tvalues[idx], fit.pvalues[idx],
                                      *fit.conf_int()[idx]] for fit in fits]).T
                self.check_maps(destination, name,
                                dict(zip(("effect", "se", "t", "p", "ci_low", "ci_high"), expected)), affine)
        self.assertFalse((self.root / "out/voxelwise/age").exists())
        # A rerun selecting another term must not delete earlier predictor maps.
        config["model"]["primary_predictors"] = "exposure"
        config["analysis"]["analysis"] = "voxelwise"
        config["references"] = {"exposure": "no"}
        run_analysis(config, prepared)
        self.assertTrue((self.root / "out/voxelwise/group/group_F.nii.gz").is_file())
        self.assertTrue((self.root / "out/voxelwise/interaction/coefficient_03/interaction_t.nii.gz").is_file())
        self.assertTrue((self.root / "out/roi/group/roi_results.csv").is_file())

        config["model"] = {"primary_predictor": "intercept"}
        config["references"] = {}
        run_analysis(config, prepared)
        expected = np.array([
            [fit.params[0], fit.bse[0], fit.tvalues[0], fit.pvalues[0], *fit.conf_int()[0]]
            for fit in (sm.OLS(y, self.ones[:, None]).fit() for y in self.y.T)
        ]).T
        self.check_maps(self.root / "out/voxelwise", "intercept",
                        dict(zip(("effect", "se", "t", "p", "ci_low", "ci_high"), expected)), affine)

    def check_maps(self, folder, name, expected, affine):
        def load(suffix, outside):
            image = nib.load(folder / f"{name}_{suffix}.nii.gz")
            np.testing.assert_array_equal(image.affine, affine)
            values = image.get_fdata().ravel()
            np.testing.assert_array_equal(values[6:], outside)
            return values[:6]

        for suffix, values in expected.items():
            np.testing.assert_allclose(load(suffix, 1 if suffix == "p" else 0),
                                       values, rtol=1e-5, atol=1e-10)
        q = multipletests(expected["p"], method="fdr_bh")[1]
        np.testing.assert_allclose(load("q_fdr", 1), q, rtol=1e-5, atol=1e-10)
        np.testing.assert_array_equal(load("fdr_q05_mask", 0), q < .05)
        np.testing.assert_array_equal(load("uncorrected_p05_mask", 0), np.asarray(expected["p"]) < .05)

    def test_invalid_primary_configuration(self):
        for model in ({}, {"primary_predictor": "age", "primary_predictors": "age, score"},
                      {"primary_predictors": " "}, {"primary_predictors": "age,"},
                      {"primary_predictors": "age, AGE"}, {"primary_predictors": "age", "covariates": "age"},
                      {"primary_predictors": "intercept, age"}, {"primary_predictor": "intercept", "covariates": "age"}):
            with self.subTest(model=model), self.assertRaises(ValueError):
                model_predictors(model)

    def test_invalid_design(self):
        with self.assertRaisesRegex(ValueError, "reference.*absent"):
            self.roi_run({"primary_predictors": "group, age"}, {"group": "absent"})
        with self.assertRaisesRegex(ValueError, "missing required model columns"):
            self.roi_run({"primary_predictors": "age, nonexistent"}, {})
        self.data["copy"] = self.data.age
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            self.roi_run({"primary_predictors": "age, copy"}, {})
        self.data.loc[0, "score"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing score"):
            self.roi_run({"primary_predictors": "age, score"}, {})
        result = self.roi_run({"primary_predictors": "age, score", "missing": "drop"}, {})
        self.assertEqual(len(result["design"]), len(self.data) - 1)
        self.data = self.data.iloc[:2]
        with self.assertRaisesRegex(ValueError, "no residual degrees of freedom"):
            self.roi_run({"primary_predictor": "age"}, {})
