"""
test_pipeline.py — Tests d'intégration pour PhotoEditorPipeline
"""

import os
import tempfile

import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from photo_editor.pipeline import PhotoEditorPipeline
from photo_editor.spatial_field import SpatialField


def _make_image(h=64, w=64):
    return np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)


def _make_depth(h=64, w=64):
    return np.random.rand(h, w).astype(np.float32)


def _make_mask(h=64, w=64, region=(10, 30, 10, 30)):
    m = np.zeros((h, w), dtype=bool)
    y1, y2, x1, x2 = region
    m[y1:y2, x1:x2] = True
    return m


def _make_crop(h=20, w=20):
    return np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)


class MockRouter:
    """Remplace HybridRouter pour éviter les appels réseau/ollama en test."""
    def route(self, prompt, confidence, material_class="surface_generique"):
        return {
            "param": "color_delta",
            "delta": 0.05,
            "raison": "test mock",
            "_model": "mock",
        }


class TestPipelineInit:
    def test_init_creates_modules(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pipeline = PhotoEditorPipeline(_make_image(), _make_depth(), memory_path=path)
            assert pipeline.spatial_field is not None
            assert pipeline.scene_graph is not None
            assert pipeline.visual_memory is not None
        finally:
            os.unlink(path)


class TestAddObject:
    def test_add_object_returns_signature(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pipeline = PhotoEditorPipeline(_make_image(), _make_depth(), memory_path=path)
            mask = _make_mask()
            sig = pipeline.add_object("obj_1", mask, (10, 10, 30, 30), _make_crop())
            assert "material_class" in sig
            assert "brightness" in sig
        finally:
            os.unlink(path)

    def test_add_object_registered_in_graph(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pipeline = PhotoEditorPipeline(_make_image(), _make_depth(), memory_path=path)
            pipeline.add_object("obj_1", _make_mask(), (10, 10, 30, 30), _make_crop())
            assert pipeline.scene_graph.get_node("obj_1") is not None
        finally:
            os.unlink(path)


class TestProcessModification:
    def _make_pipeline(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        pipeline = PhotoEditorPipeline(_make_image(), _make_depth(), memory_path=path)
        # Remplace le router par un mock pour éviter les appels réseau
        pipeline.hybrid_router = MockRouter()
        mask = _make_mask()
        pipeline.add_object("obj_1", mask, (10, 10, 30, 30), _make_crop())
        return pipeline, path

    def test_returns_expected_keys(self):
        pipeline, path = self._make_pipeline()
        try:
            mask = _make_mask()
            result = pipeline.process_modification(
                _make_image(), "obj_1", mask, _make_crop(),
                delta_params={"color_delta": 0.1},
                mask_moved=False,
            )
            assert "image" in result
            assert "impacts" in result
            assert "gain" in result
            assert "scene_hash" in result
            assert "correction" in result
        finally:
            os.unlink(path)

    def test_image_output_same_shape(self):
        pipeline, path = self._make_pipeline()
        try:
            img = _make_image()
            mask = _make_mask()
            result = pipeline.process_modification(
                img, "obj_1", mask, _make_crop(),
                delta_params={"color_delta": 0.05},
            )
            assert result["image"].shape == img.shape
        finally:
            os.unlink(path)

    def test_scene_hash_string(self):
        pipeline, path = self._make_pipeline()
        try:
            result = pipeline.process_modification(
                _make_image(), "obj_1", _make_mask(), _make_crop(),
                delta_params={},
            )
            assert isinstance(result["scene_hash"], str)
            assert len(result["scene_hash"]) == 32  # MD5
        finally:
            os.unlink(path)

    def test_mask_moved_activates_inpainter(self):
        """Vérifie que CavityInpainter est bien appelé quand mask_moved=True."""
        pipeline, path = self._make_pipeline()
        inpainter_called = []

        original_fill = pipeline.cavity_inpainter.fill
        def mock_fill(image, cavity_mask):
            inpainter_called.append(True)
            return original_fill(image, cavity_mask)

        pipeline.cavity_inpainter.fill = mock_fill
        try:
            pipeline.process_modification(
                _make_image(), "obj_1", _make_mask(), _make_crop(),
                delta_params={},
                mask_moved=True,
            )
            assert len(inpainter_called) > 0, (
                "RÈGLE 7 VIOLÉE : CavityInpainter non appelé malgré mask_moved=True"
            )
        finally:
            os.unlink(path)

    def test_spatial_field_propagated_after_modification(self):
        """Vérifie que SpatialField.propagate_modification() est appelé."""
        pipeline, path = self._make_pipeline()
        propagated = []

        original_propagate = pipeline.spatial_field.propagate_modification
        def mock_propagate(object_mask, delta_params, sigma=15.0):
            propagated.append(True)
            return original_propagate(object_mask, delta_params, sigma)

        pipeline.spatial_field.propagate_modification = mock_propagate
        try:
            pipeline.process_modification(
                _make_image(), "obj_1", _make_mask(), _make_crop(),
                delta_params={"color_delta": 0.1},
            )
            assert len(propagated) > 0, (
                "RÈGLE 1 VIOLÉE : SpatialField.propagate_modification() non appelé"
            )
        finally:
            os.unlink(path)

    def test_correction_stored_in_memory(self):
        """Vérifie que chaque correction est sauvegardée dans VisualMemory."""
        pipeline, path = self._make_pipeline()
        try:
            initial_count = len(pipeline.visual_memory._records)
            pipeline.process_modification(
                _make_image(), "obj_1", _make_mask(), _make_crop(),
                delta_params={"color_delta": 0.1},
            )
            assert len(pipeline.visual_memory._records) > initial_count
        finally:
            os.unlink(path)
