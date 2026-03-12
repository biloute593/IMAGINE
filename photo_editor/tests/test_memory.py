"""Tests for VisualMemory module."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from photo_editor.spatial_field import SpatialField
from photo_editor.visual_memory import VisualMemory


@pytest.fixture
def tmp_memory(tmp_path):
    """Return a VisualMemory backed by a temp file."""
    return VisualMemory(memory_path=tmp_path / "test_memory.json")


@pytest.fixture
def sample_crop():
    return np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 255
    return mask


@pytest.fixture
def sample_depth():
    return np.random.rand(30, 30).astype(np.float32)


@pytest.fixture
def sample_sf():
    return SpatialField(30, 30)


class TestTextureEntropy:
    def test_returns_float(self, sample_crop, sample_mask):
        result = VisualMemory.compute_texture_entropy(sample_crop, sample_mask)
        assert isinstance(result, float)

    def test_zero_for_empty_mask(self, sample_crop):
        mask = np.zeros((30, 30), dtype=np.uint8)
        result = VisualMemory.compute_texture_entropy(sample_crop, mask)
        assert result == 0.0


class TestExtractSignature:
    def test_returns_8d_dict(self, tmp_memory, sample_crop, sample_mask, sample_depth, sample_sf):
        sig = tmp_memory.extract_signature(sample_crop, sample_mask, sample_depth, sample_sf)
        expected_keys = {
            "brightness", "saturation", "specularity", "roughness",
            "depth_mean", "material_class", "texture_entropy", "scene_hash",
        }
        assert expected_keys == set(sig.keys())

    def test_scene_hash_present(self, tmp_memory, sample_crop, sample_mask, sample_depth, sample_sf):
        sig = tmp_memory.extract_signature(sample_crop, sample_mask, sample_depth, sample_sf)
        assert len(sig["scene_hash"]) == 32  # MD5 hex


class TestFindSimilar:
    def test_filters_by_scene_hash(self, tmp_memory):
        """REGLE CRITIQUE : find_similar must filter by scene_hash."""
        sig = {"brightness": 0.5, "saturation": 0.1, "specularity": 0.2,
               "roughness": 0.3, "depth_mean": 0.4}
        tmp_memory._memory = [
            {"signature": sig, "scene_hash": "hash_A", "correction": {"color_delta": 0.1}},
            {"signature": sig, "scene_hash": "hash_B", "correction": {"color_delta": 0.2}},
        ]
        results = tmp_memory.find_similar(sig, "hash_A", threshold=0.5)
        assert len(results) == 1
        assert results[0]["scene_hash"] == "hash_A"

    def test_empty_when_no_hash_match(self, tmp_memory):
        sig = {"brightness": 0.5, "saturation": 0.1, "specularity": 0.2,
               "roughness": 0.3, "depth_mean": 0.4}
        tmp_memory._memory = [
            {"signature": sig, "scene_hash": "hash_X", "correction": {}},
        ]
        results = tmp_memory.find_similar(sig, "hash_Y")
        assert len(results) == 0

    def test_threshold_filtering(self, tmp_memory):
        sig_close = {"brightness": 0.5, "saturation": 0.1, "specularity": 0.2,
                     "roughness": 0.3, "depth_mean": 0.4}
        sig_far = {"brightness": 0.0, "saturation": 0.9, "specularity": 0.9,
                   "roughness": 0.0, "depth_mean": 0.0}
        tmp_memory._memory = [
            {"signature": sig_close, "scene_hash": "h1", "correction": {}},
            {"signature": sig_far, "scene_hash": "h1", "correction": {}},
        ]
        results = tmp_memory.find_similar(sig_close, "h1", threshold=0.80)
        assert all(r["_similarity"] >= 0.80 for r in results)


class TestStore:
    def test_store_and_retrieve(self, tmp_memory, sample_crop, sample_mask):
        tmp_memory.store(
            object_id="obj1",
            crop=sample_crop,
            mask=sample_mask,
            correction={"color_delta": 0.1},
            model="qwen_local",
            score_before=0.5,
            score_after=0.7,
            scene_hash="abc123",
        )
        history = tmp_memory.get_history("obj1")
        assert len(history) == 1
        assert history[0]["object_id"] == "obj1"

    def test_persistence(self, tmp_path, sample_crop, sample_mask):
        path = tmp_path / "persist_test.json"
        mem1 = VisualMemory(memory_path=path)
        mem1.store("obj1", sample_crop, sample_mask, {"x": 1}, "local", 0.5, 0.6, "h1")

        mem2 = VisualMemory(memory_path=path)
        assert len(mem2.get_history("obj1")) == 1

    def test_get_history_limit(self, tmp_memory, sample_crop, sample_mask):
        for i in range(10):
            tmp_memory.store(f"obj1", sample_crop, sample_mask, {"i": i}, "m", 0.0, 0.0, "h")
        history = tmp_memory.get_history("obj1", n=3)
        assert len(history) == 3
