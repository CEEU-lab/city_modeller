import os
from unittest import TestCase
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points

from city_modeller.utils import (
    bound_multipol_by_bbox,
    distancia_mas_cercano,
    geometry_centroid,
    init_package,
    pob_a_distancia,
)
from tests import PROJECT_DIR

RADIOS_TEST = gpd.read_file(f"{PROJECT_DIR}/data/radios_test.geojson")
PARQUES_TEST = gpd.read_file(f"{PROJECT_DIR}/data/public_space_test.geojson")


class TestUtils(TestCase):
    @patch("os.mkdir")
    def test_init_package(self, mkdir_mock):
        init_package(PROJECT_DIR)
        mkdir_mock.assert_called_once_with(os.path.join(PROJECT_DIR, "figures"))

    def test_geometry_centroid(self):
        centroids = RADIOS_TEST.geometry.centroid
        pd.testing.assert_series_equal(geometry_centroid(RADIOS_TEST), centroids)

    def test_distancia_mas_cercano(self):
        points = [Point([0, i]) for i in range(3)]
        target_points = MultiPoint([[0.0, 0.0], [1.0, 2.0]])
        pairs = [nearest_points(point, target_points) for point in points]
        self.assertListEqual(
            [a.distance(b) for a, b in pairs],
            [distancia_mas_cercano(point, target_points) for point in points],
        )

    def test_pob_a_distancia(self):
        distancias = pd.Series([1000, 2000, 3000, 4000])
        self.assertEqual(75, pob_a_distancia(distancias, 40))

    def test_bound_multipol_by_bbox(self):
        bbox = np.array([-58.50252591, -34.70529314, -58.42404114, -34.65082308])
        bb_polygon = Polygon(
            [
                (bbox[0], bbox[3]),
                (bbox[2], bbox[3]),
                (bbox[2], bbox[1]),
                (bbox[0], bbox[1]),
            ]
        )
        gdf2 = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=["geometry"])
        pd.testing.assert_frame_equal(
            gpd.overlay(gdf2, PARQUES_TEST, how="intersection"),
            bound_multipol_by_bbox(PARQUES_TEST, bbox),
        )
