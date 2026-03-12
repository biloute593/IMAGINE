"""Tests for SpatialField module."""

import hashlib

import numpy as np
import pytest

from photo_editor.spatial_field import SpatialField


class TestSpatialFieldInit:
    def test_dimensions(self):
        sf = SpatialField(100, 200)
        assert sf.height == 100
        assert sf.width == 200

    def test_fields_initialized_to_zero(self):
        sf = SpatialField(10, 10)
        assert sf.depth_field.shape == (10, 10)
        assert sf.light_field.shape == (10, 10)
        assert sf.material_field.shape == (10, 10)
        assert sf.reflectivity_field.shape == (10, 10)
        assert sf.diffusion_field.shape == (10, 10)
        assert np.all(sf.depth_field == 0)

    def test_field_dtypes(self):
        sf = SpatialField(5, 5)
        assert sf.depth_field.dtype == np.float32
        assert sf.material_field.dtype == np.int8


class TestBuildFromScene:
    def test_build_sets_depth(self):
        sf = SpatialField(10, 10)
        depth = np.random.rand(10, 10).astype(np.float32) + 0.1
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        segments = [{"mask": mask, "bbox": (2, 2, 3, 3)}]
        signatures = [{"material_class": "metal_speculaire", "specularity": 0.8, "roughness": 0.2}]
        sf.build_from_scene(depth, segments, signatures)
        assert np.allclose(sf.depth_field, depth)

    def test_build_sets_material(self):
        sf = SpatialField(10, 10)
        depth = np.ones((10, 10), dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:3, 0:3] = True
        segments = [{"mask": mask, "bbox": (0, 0, 3, 3)}]
        signatures = [{"material_class": "verre_texture", "specularity": 0.5, "roughness": 0.3}]
        sf.build_from_scene(depth, segments, signatures)
        assert sf.material_field[1, 1] == 2  # verre_texture → 2


class TestSample:
    def test_sample_returns_dict(self):
        sf = SpatialField(5, 5)
        sf.depth_field[2, 3] = 0.5
        result = sf.sample(3, 2)
        assert isinstance(result, dict)
        assert result["depth"] == pytest.approx(0.5)

    def test_sample_out_of_bounds(self):
        sf = SpatialField(5, 5)
        with pytest.raises(IndexError):
            sf.sample(10, 10)


class TestPropagateModification:
    def test_propagation_changes_light_field(self):
        sf = SpatialField(20, 20)
        sf.light_field[:] = 0.5
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        sf.propagate_modification(mask, {"color_delta": 0.5}, sigma=5.0)
        # Center should be brighter
        assert sf.light_field[10, 10] > 0.5

    def test_sigma_capped_at_50(self):
        sf = SpatialField(10, 10)
        sf.light_field[:] = 0.5
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:6, 4:6] = True
        # Should not raise even with sigma > 50
        sf.propagate_modification(mask, {"color_delta": 0.1}, sigma=100.0)

    def test_reflectivity_propagation(self):
        sf = SpatialField(20, 20)
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        sf.propagate_modification(mask, {"reflectivity_delta": 0.5}, sigma=5.0)
        assert sf.reflectivity_field[10, 10] > 0.0

    def test_light_field_clamped(self):
        sf = SpatialField(10, 10)
        sf.light_field[:] = 0.9
        mask = np.ones((10, 10), dtype=bool)
        sf.propagate_modification(mask, {"color_delta": 10.0}, sigma=3.0)
        assert sf.light_field.max() <= 1.0


class TestComputeSceneHash:
    def test_hash_is_md5(self):
        sf = SpatialField(5, 5)
        h = sf.compute_scene_hash()
        assert len(h) == 32  # MD5 hex digest length

    def test_hash_changes_with_modification(self):
        sf = SpatialField(10, 10)
        h1 = sf.compute_scene_hash()
        sf.light_field[0, 0] = 1.0
        h2 = sf.compute_scene_hash()
        assert h1 != h2

    def test_same_state_same_hash(self):
        sf1 = SpatialField(5, 5)
        sf2 = SpatialField(5, 5)
        assert sf1.compute_scene_hash() == sf2.compute_scene_hash()
