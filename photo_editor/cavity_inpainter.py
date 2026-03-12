"""
CavityInpainter — Patch matching CPU pour remplir les cavités laissées par les objets déplacés.

Activé UNIQUEMENT si mask_moved=True.
Méthode : patch matching CPU depuis fond réel de l'image.
Limite  : 500 candidats max (perf CPU).

AGENT : Remplacer _patch_synthesis() par LaMa/MAT tiny quand GPU disponible.
"""

import cv2
import numpy as np
from scipy.ndimage import binary_dilation


class CavityInpainter:
    """Fill cavities left by moved objects using CPU patch matching."""

    MAX_CANDIDATES = 500  # AGENT : performance CPU, ne pas augmenter sans GPU
    BORDER_WIDTH = 20  # pixels around cavity for background sampling
    BLEND_RADIUS = 4  # edge blending radius

    def inpaint(self, image: np.ndarray, cavity_mask: np.ndarray) -> np.ndarray:
        """Fill the cavity defined by *cavity_mask* using patch matching.

        Parameters
        ----------
        image       : (H, W, 3) BGR image with cavity (black or old content).
        cavity_mask : (H, W) bool mask where True = cavity to fill.

        Returns
        -------
        Image with cavity filled.
        """
        result = image.copy()
        bool_mask = cavity_mask.astype(bool)

        if not bool_mask.any():
            return result

        bg_signature = self._sample_background_signature(image, bool_mask)
        result = self._patch_synthesis(result, bool_mask, bg_signature)
        result = self._blend_edges(result, bool_mask)
        return result

    def _sample_background_signature(
        self, image: np.ndarray, cavity_mask: np.ndarray
    ) -> np.ndarray:
        """Sample mean color from border region around the cavity."""
        border = binary_dilation(cavity_mask, iterations=self.BORDER_WIDTH) & ~cavity_mask
        if not border.any():
            return np.array([128, 128, 128], dtype=np.float32)
        pixels = image[border]
        return pixels.mean(axis=0).astype(np.float32)

    def _patch_synthesis(
        self,
        image: np.ndarray,
        cavity_mask: np.ndarray,
        bg_signature: np.ndarray,
    ) -> np.ndarray:
        """Fill cavity using patch matching from background regions.

        AGENT : En CPU, patch matching = seule option viable.
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Source: pixels outside mask in a dilated border
        source_region = binary_dilation(cavity_mask, iterations=self.BORDER_WIDTH) & ~cavity_mask

        # Collect source patches (5x5)
        patch_size = 5
        half = patch_size // 2
        source_ys, source_xs = np.where(source_region)

        if len(source_ys) == 0:
            result[cavity_mask] = bg_signature.astype(np.uint8)
            return result

        # Subsample for performance
        if len(source_ys) > self.MAX_CANDIDATES:
            indices = np.random.choice(len(source_ys), self.MAX_CANDIDATES, replace=False)
            source_ys = source_ys[indices]
            source_xs = source_xs[indices]

        # Collect source patches
        patches = []
        patch_coords = []
        for sy, sx in zip(source_ys, source_xs):
            if sy - half < 0 or sy + half >= h or sx - half < 0 or sx + half >= w:
                continue
            patch = image[sy - half:sy + half + 1, sx - half:sx + half + 1]
            patches.append(patch)
            patch_coords.append((sy, sx))

        if not patches:
            result[cavity_mask] = bg_signature.astype(np.uint8)
            return result

        patches = np.array(patches)
        patch_means = patches.reshape(len(patches), -1).mean(axis=1)

        # Fill cavity pixels
        cav_ys, cav_xs = np.where(cavity_mask)
        for cy, cx in zip(cav_ys, cav_xs):
            if cy - half < 0 or cy + half >= h or cx - half < 0 or cx + half >= w:
                result[cy, cx] = bg_signature.astype(np.uint8)
                continue
            # Find nearest patch by mean intensity
            target_mean = bg_signature.mean()
            dists = np.abs(patch_means - target_mean)
            best_idx = np.argmin(dists)
            best_sy, best_sx = patch_coords[best_idx]
            result[cy, cx] = image[best_sy, best_sx]

        return result

    def _blend_edges(self, image: np.ndarray, cavity_mask: np.ndarray) -> np.ndarray:
        """Smooth the boundary of the filled cavity (radius=4px)."""
        result = image.copy()
        edge = binary_dilation(cavity_mask, iterations=self.BLEND_RADIUS) & ~cavity_mask

        if not edge.any():
            return result

        ksize = self.BLEND_RADIUS * 2 + 1
        blurred = cv2.GaussianBlur(result, (ksize, ksize), 0)
        result[edge] = blurred[edge]
        return result
