"""
test_memory.py — Tests unitaires pour VisualMemory
"""

import os
import tempfile

import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from photo_editor.visual_memory import VisualMemory


def _make_crop(h=32, w=32, channels=3):
    return np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)


def _make_mask(h=32, w=32):
    m = np.zeros((h, w), dtype=bool)
    m[5:25, 5:25] = True
    return m


def _make_depth(h=32, w=32):
    return np.random.rand(h, w).astype(np.float32)


class FakeSpatialField:
    """Stub minimal pour tester sans dépendance sur SpatialField."""
    def compute_scene_hash(self):
        return "abc123"


class TestExtractSignature:
    def test_returns_8d_dict(self):
        vm = VisualMemory(memory_path="/tmp/test_vm_extract.json")
        crop = _make_crop()
        mask = _make_mask()
        depth = _make_depth()
        sf = FakeSpatialField()
        sig = vm.extract_signature(crop, mask, depth, sf)
        required_keys = [
            "brightness", "saturation", "specularity", "roughness",
            "depth_mean", "material_class", "texture_entropy", "scene_hash",
        ]
        for k in required_keys:
            assert k in sig, f"Clé manquante : {k}"

    def test_scene_hash_from_spatial_field(self):
        vm = VisualMemory(memory_path="/tmp/test_vm_hash.json")
        sf = FakeSpatialField()
        sig = vm.extract_signature(_make_crop(), _make_mask(), _make_depth(), sf)
        assert sig["scene_hash"] == "abc123"

    def test_brightness_in_range(self):
        vm = VisualMemory(memory_path="/tmp/test_vm_bright.json")
        # Image blanche
        crop = np.full((32, 32, 3), 255, dtype=np.uint8)
        sig = vm.extract_signature(crop, _make_mask(), _make_depth(), FakeSpatialField())
        assert 0.0 <= sig["brightness"] <= 1.0

    def test_material_class_is_valid(self):
        valid = {"metal_speculaire", "verre_texture", "fourrure_textile",
                 "plastique_mat", "surface_generique"}
        vm = VisualMemory(memory_path="/tmp/test_vm_mat.json")
        sig = vm.extract_signature(_make_crop(), _make_mask(), _make_depth(), FakeSpatialField())
        assert sig["material_class"] in valid


class TestComputeTextureEntropy:
    def test_uniform_image_low_entropy(self):
        vm = VisualMemory(memory_path="/tmp/test_vm_entropy.json")
        crop = np.full((32, 32), 128, dtype=np.uint8)
        ent = vm.compute_texture_entropy(crop)
        assert ent == 0.0  # Un seul niveau de gris → entropie nulle

    def test_random_image_higher_entropy(self):
        vm = VisualMemory(memory_path="/tmp/test_vm_entropy2.json")
        crop = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        ent = vm.compute_texture_entropy(crop)
        assert ent > 4.0  # Image aléatoire → entropie élevée

    def test_returns_float(self):
        vm = VisualMemory(memory_path="/tmp/test_vm_ent3.json")
        ent = vm.compute_texture_entropy(_make_crop())
        assert isinstance(ent, float)


class TestFindSimilar:
    def _base_sig(self, scene_hash="scene_a"):
        return {
            "brightness": 0.5,
            "saturation": 0.1,
            "specularity": 0.05,
            "roughness": 0.02,
            "depth_mean": 0.4,
            "material_class": "plastique_mat",
            "texture_entropy": 4.0,
            "scene_hash": scene_hash,
        }

    def test_find_similar_same_hash(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        vm = VisualMemory(memory_path=path)
        sig = self._base_sig("scene_a")
        vm._records.append({
            "object_id": "obj_1",
            "signature": sig,
            "correction": {"param": "color_delta", "delta": 0.1},
            "model": "test",
            "score_before": 0.5,
            "score_after": 0.7,
            "gain": 0.2,
        })
        results = vm.find_similar(sig, "scene_a", threshold=0.5)
        assert len(results) > 0
        os.unlink(path)

    def test_CRITICAL_find_similar_rejects_different_hash(self):
        """
        RÈGLE CRITIQUE : find_similar() ne doit JAMAIS retourner de corrections
        d'une scène différente, même avec une similarité > 0.85.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        vm = VisualMemory(memory_path=path)
        sig_a = self._base_sig("scene_a")
        vm._records.append({
            "object_id": "obj_1",
            "signature": sig_a,
            "correction": {"param": "color_delta", "delta": 0.1},
            "model": "test",
            "score_before": 0.5,
            "score_after": 0.99,
            "gain": 0.49,
        })
        # Cherche avec scene_hash différent
        results = vm.find_similar(sig_a, "scene_B_different", threshold=0.0)
        assert len(results) == 0, (
            "RÈGLE CRITIQUE VIOLÉE : find_similar() a retourné des corrections "
            "d'une scène différente (scene_hash mismatch)"
        )
        os.unlink(path)

    def test_find_similar_threshold(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        vm = VisualMemory(memory_path=path)
        sig = self._base_sig("scene_x")
        # Signature très différente
        different_sig = {**sig, "brightness": 1.0, "saturation": 1.0}
        vm._records.append({
            "object_id": "obj_2",
            "signature": different_sig,
            "correction": {"param": "color_delta", "delta": 0.5},
            "model": "test",
            "score_before": 0.3,
            "score_after": 0.8,
            "gain": 0.5,
        })
        results = vm.find_similar(sig, "scene_x", threshold=0.99)
        assert len(results) == 0
        os.unlink(path)


class TestStore:
    def test_store_creates_record(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        vm = VisualMemory(memory_path=path)
        vm.store(
            "obj_test", _make_crop(), _make_mask(),
            {"param": "color_delta", "delta": 0.2},
            "qwen", 0.5, 0.8, "hash_xyz"
        )
        assert len(vm._records) == 1
        assert vm._records[0]["object_id"] == "obj_test"
        os.unlink(path)

    def test_store_persists_to_disk(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        vm = VisualMemory(memory_path=path)
        vm.store(
            "obj_persist", _make_crop(), _make_mask(),
            {"param": "lumiere_ambiante", "delta": -0.1},
            "gemini", 0.6, 0.9, "hash_abc"
        )
        # Recharge depuis disque
        vm2 = VisualMemory(memory_path=path)
        assert len(vm2._records) == 1
        os.unlink(path)


class TestGetHistory:
    def test_get_history_last_n(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        vm = VisualMemory(memory_path=path)
        for i in range(7):
            vm._records.append({
                "object_id": "obj_hist",
                "correction": {"param": "color_delta", "delta": i * 0.1},
            })
        history = vm.get_history("obj_hist", n=5)
        assert len(history) == 5
        os.unlink(path)

    def test_get_history_wrong_id(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        vm = VisualMemory(memory_path=path)
        history = vm.get_history("nonexistent")
        assert history == []
        os.unlink(path)
