#!/usr/bin/env python3

import unittest

import numpy as np
import math

from src.nonconvex_clusters import nonconvex_clusters


class TestNonconvexClusters(unittest.TestCase):

    def test_size(self):
        df = nonconvex_clusters()
        self.assertEqual(
            df.shape, (4, 4),
            msg="nonconvex_clusters() should return a DataFrame with 4 rows "
                "(one per eps value) and 4 columns. Got shape %r."
                % (df.shape,))

    def test_type(self):
        df = nonconvex_clusters()
        self.assertEqual(
            list(df.dtypes.values), [float, float, float, float],
            msg="nonconvex_clusters() returned a DataFrame with incorrect "
                "column types; all four columns should be floats.")

    def test_eps(self):
        df = nonconvex_clusters()
        np.testing.assert_allclose(
            df.eps.values, [0.05, 0.10, 0.15, 0.20],
            err_msg="The 'eps' column should contain exactly "
                    "[0.05, 0.10, 0.15, 0.20], in that order.")

    def test_columns(self):
        df = nonconvex_clusters()
        self.assertEqual(
            list(df.columns.values), ["eps", "Score", "Clusters", "Outliers"],
            msg="nonconvex_clusters() returned a DataFrame with incorrect "
                "column names; expected ['eps', 'Score', 'Clusters', "
                "'Outliers'] in that order.")

    def test_scores(self):
        df = nonconvex_clusters()
        self.assertAlmostEqual(
            df.loc[1, "Score"], 1.0,
            msg="Score for eps=0.10 (row 1) should be 1.0.")
        self.assertAlmostEqual(
            df.loc[2, "Score"], 1.0,
            msg="Score for eps=0.15 (row 2) should be 1.0.")
        self.assertTrue(
            math.isnan(df.Score[0]),
            msg="Score for eps=0.05 (row 0) should be NaN: too many points "
                "end up unclustered/outliers at that eps for a silhouette "
                "score to be defined.")
        self.assertTrue(
            math.isnan(df.Score[3]),
            msg="Score for eps=0.20 (row 3) should be NaN: with only one "
                "cluster found, a silhouette score is undefined.")

    def test_clusters(self):
        df = nonconvex_clusters()
        np.testing.assert_allclose(
            df.Clusters, [12, 2, 2, 1],
            err_msg="The 'Clusters' column should contain the number of "
                    "clusters DBSCAN found at each eps: [12, 2, 2, 1].")

    def test_outliers(self):
        df = nonconvex_clusters()
        np.testing.assert_allclose(
            df.Outliers, [118, 3, 0, 0],
            err_msg="The 'Outliers' column should contain the number of "
                    "points DBSCAN labeled as noise at each eps: "
                    "[118, 3, 0, 0].")


if __name__ == '__main__':
    unittest.main()
