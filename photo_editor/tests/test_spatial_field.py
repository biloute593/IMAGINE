"""
test_spatial_field.py — Tests unitaires pour le module SpatialField
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from photo_editor.spatial_field import SpatialField, MATERIAL_NAMES


class TestSpatialFieldInit:
    def test_default_shapes(self):
        sf = SpatialField(100, 120)
        assert sf.depth_field.shape == (100, 120)
        assert sf.light_field.shape == (100, 120)
        assert sf.material_field.shape == (100, 120)
        assert sf.reflectivity_field.shape == (100, 120)
        assert sf.diffusion_field.shape == (100, 120)

    def test_dtypes(self):
        sf = SpatialField(50, 50)
        assert sf.depth_field.dtype == np.float32
        assert sf.light_field.dtype == np.float32
        assert sf.material_field.dtype == np.int8
        assert sf.reflectivity_field.dtype == np.float32
        assert sf.diffusion_field.dtype == np.float32


class TestBuildFromScene:
    def _make_sf_and_depth(self, h=64, w=64):
        sf = SpatialField(h, w)
        depth = np.random.rand(h, w).astype(np.float32)
        return sf, depth

    def test_build_no_segments(self):
        sf, depth = self._make_sf_and_depth()
        sf.build_from_scene(depth, [], [])
        # depth_field doit être normalisé en [0, 1]
        assert sf.depth_field.min() >= 0.0
        assert sf.depth_field.max() <= 1.0

    def test_build_with_segment(self):
        sf, depth = self._make_sf_and_depth()
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True
        sig = {
            "material_class": "metal_speculaire",
            "brightness": 0.9,
        }
        sf.build_from_scene(depth, [mask], [sig])
        # La zone du masque doit avoir le material_id de metal_speculaire
        assert int(sf.material_field[15, 15]) == MATERIAL_NAMES["metal_speculaire"]

    def test_build_shape_mismatch(self):
        sf = SpatialField(64, 64)
        depth_wrong = np.ones((32, 32), dtype=np.float32)
        with pytest.raises(ValueError):
            sf.build_from_scene(depth_wrong, [], [])

    def test_build_constant_depth(self):
        sf = SpatialField(32, 32)
        depth = np.ones((32, 32), dtype=np.float32) * 0.5
        sf.build_from_scene(depth, [], [])
        # Pas de division par zéro
        assert not np.any(np.isnan(sf.depth_field))


class TestSample:
    def test_sample_returns_dict(self):
        sf = SpatialField(50, 50)
        result = sf.sample(25, 25)
        assert "depth" in result
        assert "luminance" in result
        assert "material_id" in result
        assert "reflectivity" in result
        assert "light_diffusion" in result

    def test_sample_clamps_coordinates(self):
        sf = SpatialField(50, 50)
        # Coordonnées hors limites ne doivent pas lever d'exception
        sf.sample(-10, -10)
        sf.sample(1000, 1000)


class TestPropagateModification:
    def _make_sf(self):
        sf = SpatialField(64, 64)
        depth = np.random.rand(64, 64).astype(np.float32)
        sf.build_from_scene(depth, [], [])
        return sf

    def test_color_delta_propagation(self):
        sf = self._make_sf()
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:40, 20:40] = True
        light_before = sf.light_field.copy()
        sf.propagate_modification(mask, {"color_delta": 0.5}, sigma=5.0)
        # Le champ lumineux doit avoir changé
        assert not np.allclose(sf.light_field, light_before)

    def test_sigma_clamped(self):
        sf = self._make_sf()
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:40, 20:40] = True
        # sigma=200 doit être réduit à 50 sans erreur
        sf.propagate_modification(mask, {"color_delta": 0.1}, sigma=200.0)

    def test_light_field_stays_in_range(self):
        sf = self._make_sf()
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:50, 10:50] = True
        # Application répétée avec delta extrême
        for _ in range(10):
            sf.propagate_modification(mask, {"color_delta": 1.0}, sigma=15.0)
        assert sf.light_field.min() >= 0.0
        assert sf.light_field.max() <= 1.0

    def test_shadow_darkens_surroundings(self):
        sf = self._make_sf()
        mask = np.zeros((64, 64), dtype=bool)
        mask[30:34, 30:34] = True
        light_before = sf.light_field.copy()
        sf.propagate_modification(mask, {"shadow_intensity": 0.5}, sigma=10.0)
        # La luminosité globale doit avoir diminué légèrement
        assert sf.light_field.mean() <= light_before.mean() + 1e-4


class TestComputeSceneHash:
    def test_returns_md5_string(self):
        sf = SpatialField(32, 32)
        h = sf.compute_scene_hash()
        assert isinstance(h, str)
        assert len(h) == 32

    def test_hash_changes_after_modification(self):
        sf = SpatialField(32, 32)
        depth = np.random.rand(32, 32).astype(np.float32)
        sf.build_from_scene(depth, [], [])
        h1 = sf.compute_scene_hash()
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True
        sf.propagate_modification(mask, {"color_delta": 0.5})
        h2 = sf.compute_scene_hash()
        assert h1 != h2

    def test_hash_consistent_same_state(self):
        sf = SpatialField(32, 32)
        h1 = sf.compute_scene_hash()
        h2 = sf.compute_scene_hash()
        assert h1 == h2
