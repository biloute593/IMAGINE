"""
VisualMemory — Signature 8D par objet + scene_hash.
Stockage persistant dans visual_memory.json.

RÈGLE CRITIQUE — DÉSYNCHRONISATION :
  find_similar() DOIT filtrer par scene_hash compatible.
  Une correction stockée pour scène A est invalide pour scène B.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import entropy as shannon_entropy


_MEMORY_PATH = Path(__file__).parent / "visual_memory.json"


class VisualMemory:
    """8D visual memory with scene_hash filtering."""

    def __init__(self, memory_path: str | Path | None = None):
        self.memory_path = Path(memory_path) if memory_path else _MEMORY_PATH
        self._memory: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.memory_path.exists():
            with open(self.memory_path, "r") as f:
                self._memory = json.load(f)

    def _save(self) -> None:
        with open(self.memory_path, "w") as f:
            json.dump(self._memory, f, indent=2)

    # ------------------------------------------------------------------
    # Signature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def compute_texture_entropy(crop: np.ndarray, mask: np.ndarray) -> float:
        """Compute Shannon entropy of the texture inside mask."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        pixels = gray[mask.astype(bool)]
        if len(pixels) == 0:
            return 0.0
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        return float(shannon_entropy(hist, base=2))

    @staticmethod
    def _classify_material(specularity: float, roughness: float, tex_entropy: float) -> str:
        """Classify material from physical properties."""
        if specularity > 0.6 and roughness < 0.3:
            return "metal_speculaire"
        if specularity > 0.4 and tex_entropy > 4.0:
            return "verre_texture"
        if roughness > 0.7 and tex_entropy > 5.0:
            return "fourrure_textile"
        if specularity < 0.2 and roughness > 0.4:
            return "plastique_mat"
        return "surface_generique"

    def extract_signature(
        self,
        crop: np.ndarray,
        mask: np.ndarray,
        depth_map: np.ndarray,
        spatial_field,
    ) -> dict:
        """Compute 8D signature for an object region.

        Parameters
        ----------
        crop        : BGR image patch of the object.
        mask        : binary mask of the object (may be full-image or crop-sized).
        depth_map   : full-scene depth map from MiDaS.
        spatial_field : SpatialField instance (for scene_hash).
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        ch, cw = gray.shape[:2]

        # Resize mask to crop dimensions when they differ
        if mask.shape[:2] != (ch, cw):
            crop_mask = cv2.resize(mask.astype(np.uint8), (cw, ch)) > 0
        else:
            crop_mask = mask.astype(bool)

        bool_mask = crop_mask
        pixels = gray[bool_mask] if bool_mask.any() else gray.ravel()

        brightness = float(np.mean(pixels) / 255.0)

        if crop.ndim == 3:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            sat_pixels = hsv[:, :, 1][bool_mask] if bool_mask.any() else hsv[:, :, 1].ravel()
            saturation = float(np.std(sat_pixels) / 255.0)
        else:
            saturation = 0.0

        bright_threshold = np.percentile(pixels, 95) if len(pixels) > 0 else 255
        specularity = float(np.mean(pixels > bright_threshold)) if len(pixels) > 0 else 0.0

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        roughness = float(np.mean(gradient[bool_mask]) / 255.0) if bool_mask.any() else 0.0

        # Depth mean — resize depth_map region to match crop dimensions
        dh, dw = depth_map.shape[:2]
        if (ch, cw) != (dh, dw):
            depth_region = cv2.resize(depth_map, (cw, ch))
        else:
            depth_region = depth_map
        depth_mean = float(np.mean(depth_region[bool_mask])) if bool_mask.any() else 0.0

        texture_entropy = self.compute_texture_entropy(crop, bool_mask.astype(np.uint8))

        material_class = self._classify_material(specularity, roughness, texture_entropy)

        scene_hash = spatial_field.compute_scene_hash() if spatial_field else ""

        return {
            "brightness": brightness,
            "saturation": saturation,
            "specularity": specularity,
            "roughness": roughness,
            "depth_mean": depth_mean,
            "material_class": material_class,
            "texture_entropy": texture_entropy,
            "scene_hash": scene_hash,
        }

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    @staticmethod
    def _signature_distance(a: dict, b: dict) -> float:
        """Euclidean distance on the 5 numeric signature dimensions."""
        keys = ["brightness", "saturation", "specularity", "roughness", "depth_mean"]
        return float(np.sqrt(sum((a.get(k, 0) - b.get(k, 0)) ** 2 for k in keys)))

    def find_similar(
        self,
        signature: dict,
        scene_hash: str,
        threshold: float = 0.80,
    ) -> list[dict]:
        """Find corrections with similar signature AND compatible scene_hash.

        AGENT : Ne jamais désactiver le filtre scene_hash. Bug #1 du projet.
        """
        results = []
        for entry in self._memory:
            # RÈGLE CRITIQUE : filtrer par scene_hash
            if entry.get("scene_hash", "") != scene_hash:
                continue
            dist = self._signature_distance(signature, entry.get("signature", {}))
            similarity = max(0.0, 1.0 - dist)
            if similarity >= threshold:
                results.append({**entry, "_similarity": similarity})
        results.sort(key=lambda x: x["_similarity"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        object_id: str,
        crop: np.ndarray,
        mask: np.ndarray,
        correction: dict,
        model: str,
        score_before: float,
        score_after: float,
        scene_hash: str,
        signature: dict | None = None,
    ) -> None:
        """Persist a correction with its context."""
        entry = {
            "object_id": object_id,
            "correction": correction,
            "model": model,
            "score_before": score_before,
            "score_after": score_after,
            "scene_hash": scene_hash,
            "signature": signature or {},
        }
        self._memory.append(entry)
        self._save()

    def get_history(self, object_id: str, n: int = 5) -> list[dict]:
        """Return last *n* corrections for *object_id*."""
        return [e for e in self._memory if e["object_id"] == object_id][-n:]
