"""
SpatialField — Représentation continue de la scène.
Chaque pixel (x,y) possède : depth, luminance, material_id, reflectivity, light_diffusion.
Remplace les effets physiques du SceneGraph (qui garde uniquement les relations topologiques).
"""

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter


class SpatialField:
    """Continuous spatial field for light/material/depth per pixel."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.depth_field = np.zeros((height, width), dtype=np.float32)
        self.light_field = np.zeros((height, width), dtype=np.float32)
        self.material_field = np.zeros((height, width), dtype=np.int8)
        self.reflectivity_field = np.zeros((height, width), dtype=np.float32)
        self.diffusion_field = np.zeros((height, width), dtype=np.float32)

    def build_from_scene(
        self,
        depth_map: np.ndarray,
        segments: list[dict],
        signatures: list[dict],
    ) -> None:
        """Initialise tous les champs depuis MiDaS + SAM + signatures 8D.

        Parameters
        ----------
        depth_map : (H, W) float32 depth from MiDaS.
        segments  : list of dicts with keys ``mask`` (H,W bool) and ``bbox``.
        signatures: list of 8D signature dicts aligned with *segments*.
        """
        self.depth_field = depth_map.astype(np.float32)

        # Derive base luminance from depth (simple inverse-square approximation)
        depth_safe = np.clip(self.depth_field, 0.01, None)
        self.light_field = (1.0 / depth_safe)
        self.light_field /= self.light_field.max() + 1e-8  # normalize 0-1

        # AGENT : material_class mapping used across project
        _material_map = {
            "metal_speculaire": 1,
            "verre_texture": 2,
            "fourrure_textile": 3,
            "plastique_mat": 4,
            "surface_generique": 0,
        }

        for seg, sig in zip(segments, signatures):
            mask = seg["mask"].astype(bool)
            mat_name = sig.get("material_class", "surface_generique")
            self.material_field[mask] = _material_map.get(mat_name, 0)
            self.reflectivity_field[mask] = sig.get("specularity", 0.0)
            self.diffusion_field[mask] = 1.0 - sig.get("roughness", 0.5)

    def sample(self, x: int, y: int) -> dict:
        """Return physical properties of pixel (x, y)."""
        if not (0 <= y < self.height and 0 <= x < self.width):
            raise IndexError(f"Pixel ({x}, {y}) out of bounds ({self.width}x{self.height})")
        return {
            "depth": float(self.depth_field[y, x]),
            "luminance": float(self.light_field[y, x]),
            "material_id": int(self.material_field[y, x]),
            "reflectivity": float(self.reflectivity_field[y, x]),
            "light_diffusion": float(self.diffusion_field[y, x]),
        }

    def propagate_modification(
        self,
        object_mask: np.ndarray,
        delta_params: dict,
        sigma: float = 15.0,
    ) -> None:
        """Propage impact lumière/reflet/ombre avec décroissance gaussienne.

        Parameters
        ----------
        object_mask : (H, W) bool mask of the modified object.
        delta_params: dict of parameter deltas (e.g. color_delta, reflectivity_delta).
        sigma       : Gaussian decay radius in pixels. Default 15, max 50.
        """
        # AGENT : sigma=15.0 par défaut. Ne pas dépasser 50.
        sigma = min(sigma, 50.0)

        influence = object_mask.astype(np.float32)
        influence = gaussian_filter(influence, sigma=sigma)
        influence = np.clip(influence, 0.0, 1.0)

        if "color_delta" in delta_params:
            self.light_field += influence * delta_params["color_delta"] * 0.3

        if "reflectivity_delta" in delta_params:
            self.reflectivity_field += influence * delta_params["reflectivity_delta"] * 0.5

        if "shadow_intensity" in delta_params:
            self.light_field -= influence * delta_params["shadow_intensity"] * 0.2

        if "lumiere_ambiante" in delta_params:
            self.light_field += influence * delta_params["lumiere_ambiante"] * 0.4

        self.light_field = np.clip(self.light_field, 0.0, 1.0)
        self.reflectivity_field = np.clip(self.reflectivity_field, 0.0, 1.0)

    def compute_scene_hash(self) -> str:
        """Hash MD5 du light_field + depth_field (état physique de la scène).

        Utilisé par VisualMemory pour invalider corrections obsolètes.
        """
        data = self.light_field.tobytes() + self.depth_field.tobytes()
        return hashlib.md5(data).hexdigest()
