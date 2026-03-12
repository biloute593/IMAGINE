"""
cavity_inpainter.py — CavityInpainter par patch matching CPU
Remplit le vide laissé par un objet déplacé en utilisant des patches
prélevés sur le fond de l'image.

AGENT : Activé UNIQUEMENT si mask_moved=True dans process_modification().
        Remplacer _patch_synthesis() par LaMa/MAT tiny quand GPU disponible.
        En CPU, patch matching = seule option viable.

Limites :
  - 500 candidats max (performance CPU)
  - Source patches : bordure de 20px autour de la cavité
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BORDER_WIDTH = 20       # pixels de bordure autour de la cavité pour chercher patches
MAX_CANDIDATES = 500    # AGENT : ne pas dépasser sans GPU
PATCH_SIZE = 7          # taille des patches (impair)
BLEND_RADIUS = 4        # rayon de correction des bords


class CavityInpainter:
    """
    Remplit une cavité (zone vide après déplacement d'objet) par patch matching
    à partir du fond réel de l'image.

    Séquence :
      1. _sample_background_signature() — caractérise la texture autour
      2. _patch_synthesis()             — remplissage par patch matching
      3. _blend_edges()                 — correction de gradient aux bords
    """

    def fill(
        self,
        image: np.ndarray,
        cavity_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Remplit la cavité dans l'image.

        Args:
            image        : image (H, W, 3) uint8.
            cavity_mask  : masque binaire (H, W) bool/uint8 de la cavité à remplir.

        Returns:
            Image avec la cavité remplie.
        """
        mask_bool = cavity_mask.astype(bool)
        if not mask_bool.any():
            return image.copy()

        # 1. Signature du fond autour de la cavité
        bg_sig = self._sample_background_signature(image, mask_bool)
        logger.debug("Fond autour cavité : brightness=%.3f", bg_sig.get("brightness", 0))

        # 2. Remplissage par patch matching
        filled = self._patch_synthesis(image, mask_bool, bg_sig)

        # 3. Correction gradient aux bords
        result = self._blend_edges(filled, mask_bool)

        return result

    # ------------------------------------------------------------------
    # Étape 1 : signature du fond
    # ------------------------------------------------------------------

    def _sample_background_signature(
        self, image: np.ndarray, cavity_mask: np.ndarray
    ) -> dict:
        """
        Calcule la signature texturale du fond autour de la cavité.

        Args:
            image        : image source.
            cavity_mask  : masque de la cavité.

        Returns:
            dict avec brightness, color_mean, texture_std.
        """
        h, w = image.shape[:2]
        kernel = np.ones((BORDER_WIDTH * 2 + 1, BORDER_WIDTH * 2 + 1), np.uint8)
        dilated = cv2.dilate(cavity_mask.astype(np.uint8), kernel)
        border_zone = (dilated.astype(bool)) & (~cavity_mask)

        if not border_zone.any():
            # Fallback : utilise toute l'image hors cavité
            border_zone = ~cavity_mask

        bg_pixels = image[border_zone].astype(np.float32)
        brightness = float(np.mean(bg_pixels) / 255.0)
        color_mean = bg_pixels.mean(axis=0) if bg_pixels.ndim > 1 else np.array([128.0] * 3)
        texture_std = float(np.std(bg_pixels))

        return {
            "brightness": brightness,
            "color_mean": color_mean,
            "texture_std": texture_std,
            "border_zone": border_zone,
        }

    # ------------------------------------------------------------------
    # Étape 2 : synthèse par patch matching
    # ------------------------------------------------------------------

    def _patch_synthesis(
        self,
        image: np.ndarray,
        cavity_mask: np.ndarray,
        bg_sig: dict,
    ) -> np.ndarray:
        """
        Remplit la cavité pixel par pixel en cherchant le meilleur patch
        dans la bordure de fond.

        AGENT : MAX_CANDIDATES=500 pour les performances CPU.
        """
        result = image.copy()
        h, w = image.shape[:2]
        half = PATCH_SIZE // 2

        border_zone = bg_sig.get("border_zone", ~cavity_mask)

        # Collecte les coordonnées candidates dans la bordure
        ys_bg, xs_bg = np.where(border_zone)
        if len(xs_bg) == 0:
            return result

        # Limite à MAX_CANDIDATES positions sources
        if len(xs_bg) > MAX_CANDIDATES:
            idx = np.random.choice(len(xs_bg), MAX_CANDIDATES, replace=False)
            xs_bg = xs_bg[idx]
            ys_bg = ys_bg[idx]

        # Coordonnées de la cavité à remplir (parcours en raster scan)
        ys_cav, xs_cav = np.where(cavity_mask)

        for cy, cx in zip(ys_cav, xs_cav):
            # Patch autour du pixel à remplir (depuis les bords de la cavité)
            best_val = self._find_best_patch_value(
                image, cx, cy, xs_bg, ys_bg, h, w, half
            )
            result[cy, cx] = best_val

        return result

    def _find_best_patch_value(
        self,
        image: np.ndarray,
        cx: int,
        cy: int,
        xs_src: np.ndarray,
        ys_src: np.ndarray,
        h: int,
        w: int,
        half: int,
    ) -> np.ndarray:
        """
        Trouve le pixel de fond le plus similaire à la voisinage de (cx, cy).
        Utilise la différence moyenne absolue sur un voisinage 3×3.
        """
        # Voisinage de la cible (pixels hors cavité)
        x1, x2 = max(0, cx - 1), min(w, cx + 2)
        y1, y2 = max(0, cy - 1), min(h, cy + 2)
        target_region = image[y1:y2, x1:x2].astype(np.float32)

        best_dist = float("inf")
        best_pixel = image[
            np.clip(ys_src[0], 0, h - 1),
            np.clip(xs_src[0], 0, w - 1),
        ]

        for sx, sy in zip(xs_src, ys_src):
            sx, sy = int(sx), int(sy)
            sx1, sx2 = max(0, sx - 1), min(w, sx + 2)
            sy1, sy2 = max(0, sy - 1), min(h, sy + 2)
            src_region = image[sy1:sy2, sx1:sx2].astype(np.float32)

            # Ajuste les tailles pour comparaison
            rh = min(target_region.shape[0], src_region.shape[0])
            rw = min(target_region.shape[1], src_region.shape[1])
            if rh == 0 or rw == 0:
                continue

            dist = float(np.mean(np.abs(
                target_region[:rh, :rw] - src_region[:rh, :rw]
            )))
            if dist < best_dist:
                best_dist = dist
                best_pixel = image[sy, sx]

        return best_pixel

    # ------------------------------------------------------------------
    # Étape 3 : correction gradient bords
    # ------------------------------------------------------------------

    def _blend_edges(
        self, image: np.ndarray, cavity_mask: np.ndarray
    ) -> np.ndarray:
        """
        Lisse les discontinuités aux bords de la cavité remplie.
        Utilise un lissage gaussien localisé sur le périmètre.
        """
        kernel = np.ones((BLEND_RADIUS * 2 + 1, BLEND_RADIUS * 2 + 1), np.uint8)
        edge_zone = cv2.dilate(cavity_mask.astype(np.uint8), kernel)
        edge_zone = edge_zone.astype(bool) & ~cavity_mask

        # Applique un léger blur sur les bords de la cavité
        blurred = cv2.GaussianBlur(image, (BLEND_RADIUS * 2 + 1, BLEND_RADIUS * 2 + 1), 0)
        result = image.copy()
        result[edge_zone] = blurred[edge_zone]

        return result
