"""Tests for the full PhotoEditorPipeline."""

import numpy as np
import pytest

from photo_editor.pipeline import PhotoEditorPipeline


@pytest.fixture
def simple_pipeline():
    image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    depth = np.random.rand(100, 100).astype(np.float32) + 0.1
    return PhotoEditorPipeline(image, depth)


@pytest.fixture
def mask_50x50():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    return mask


class TestAddObject:
    def test_add_object_registers(self, simple_pipeline, mask_50x50):
        crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        simple_pipeline.add_object("obj1", mask_50x50, (25, 25, 50, 50), crop)
        assert "obj1" in simple_pipeline._objects
        assert "obj1" in simple_pipeline.scene_graph.nodes

    def test_add_multiple_objects(self, simple_pipeline):
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[10:30, 10:30] = 255
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[60:80, 60:80] = 255
        crop1 = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        crop2 = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        simple_pipeline.add_object("a", mask1, (10, 10, 20, 20), crop1)
        simple_pipeline.add_object("b", mask2, (60, 60, 20, 20), crop2)
        assert len(simple_pipeline._objects) == 2


class TestProcessModification:
    def test_returns_expected_keys(self, simple_pipeline, mask_50x50):
        crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        simple_pipeline.add_object("obj1", mask_50x50, (25, 25, 50, 50), crop)
        result = simple_pipeline.process_modification(
            image=simple_pipeline.image,
            object_id="obj1",
            mask=mask_50x50,
            crop=crop,
            delta_params={"color_delta": 0.1},
        )
        assert "image" in result
        assert "impacts" in result
        assert "gain" in result
        assert "scene_hash" in result

    def test_image_shape_preserved(self, simple_pipeline, mask_50x50):
        crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        simple_pipeline.add_object("obj1", mask_50x50, (25, 25, 50, 50), crop)
        result = simple_pipeline.process_modification(
            image=simple_pipeline.image,
            object_id="obj1",
            mask=mask_50x50,
            crop=crop,
            delta_params={"color_delta": 0.1},
        )
        assert result["image"].shape == simple_pipeline.image.shape

    def test_scene_hash_is_string(self, simple_pipeline, mask_50x50):
        crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        simple_pipeline.add_object("obj1", mask_50x50, (25, 25, 50, 50), crop)
        result = simple_pipeline.process_modification(
            image=simple_pipeline.image,
            object_id="obj1",
            mask=mask_50x50,
            crop=crop,
            delta_params={"color_delta": 0.1},
        )
        assert isinstance(result["scene_hash"], str)
        assert len(result["scene_hash"]) == 32

    def test_mask_moved_triggers_inpainting(self, simple_pipeline, mask_50x50):
        crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        simple_pipeline.add_object("obj1", mask_50x50, (25, 25, 50, 50), crop)
        # Should not raise with mask_moved=True
        result = simple_pipeline.process_modification(
            image=simple_pipeline.image,
            object_id="obj1",
            mask=mask_50x50,
            crop=crop,
            delta_params={"position_delta": (10, 10)},
            mask_moved=True,
        )
        assert result["image"] is not None

    def test_memory_stores_correction(self, simple_pipeline, mask_50x50):
        crop = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        simple_pipeline.add_object("obj1", mask_50x50, (25, 25, 50, 50), crop)
        simple_pipeline.process_modification(
            image=simple_pipeline.image,
            object_id="obj1",
            mask=mask_50x50,
            crop=crop,
            delta_params={"color_delta": 0.2},
        )
        history = simple_pipeline.memory.get_history("obj1")
        assert len(history) >= 1
