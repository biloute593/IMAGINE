"""
modification_engine.py — ObjectModificationEngine
Applique les corrections LLM sur les régions masquées de l'image.

Paramètres applicables :
  position_delta      (x, y) déplacement en pixels
  scale_delta         float  facteur d'échelle
  rotation_delta      float  angle en degrés
  color_delta         float  shift luminosité/teinte
  texture_delta       float  modification rugosité
  reflectivity_delta  float  modification réflectivité
  lumiere_ambiante    float  correction lumière globale
  shadow_intensity    float  intensité ombre portée
  refraction          float  indice réfraction (verre uniquement)

AGENT : Vérifier material_class avant d'appliquer refraction
        (uniquement verre_texture) ou reflectivity_delta (metal + verre).
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ObjectModificationEngine:
    """
    Applique une correction du LLM sur la région d'un objet dans l'image.
    """

    # ------------------------------------------------------------------
    # Application principale
    # ------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        correction: dict,
        material_class: str = "surface_generique",
    ) -> np.ndarray:
        """
        Applique UNE correction sur la région du masque.

        Args:
            image          : image (H, W, 3) uint8.
            mask           : masque binaire (H, W) bool/uint8.
            correction     : dict issu du LLM {"param", "delta", ...}.
            material_class : classe matériau de l'objet.

        Returns:
            image modifiée (H, W, 3) uint8.
        """
        result = image.copy()
        mask_bool = mask.astype(bool)

        param = correction.get("param", "")
        delta = float(correction.get("delta", 0.0))

        if param == "color_delta":
            result = self._apply_color_delta(result, mask_bool, delta)

        elif param == "lumiere_ambiante":
            result = self._apply_brightness(result, mask_bool, delta)

        elif param == "reflectivity_delta":
            # AGENT : uniquement metal_speculaire et verre_texture
            if material_class in ("metal_speculaire", "verre_texture"):
                result = self._apply_reflectivity(result, mask_bool, delta)
            else:
                logger.info(
                    "reflectivity_delta ignoré pour matériau '%s'", material_class
                )

        elif param == "texture_delta":
            result = self._apply_texture(result, mask_bool, delta)

        elif param == "shadow_intensity":
            result = self._apply_shadow(result, mask_bool, delta)

        elif param == "refraction":
            # AGENT : uniquement verre_texture
            if material_class == "verre_texture":
                result = self._apply_refraction(result, mask_bool, delta)
            else:
                logger.info("refraction ignorée pour matériau '%s'", material_class)

        elif param == "scale_delta":
            result = self._apply_scale(result, mask_bool, delta)

        elif param == "rotation_delta":
            result = self._apply_rotation(result, mask_bool, delta)

        elif param == "position_delta":
            # position_delta est géré au niveau pipeline (déplacement du masque)
            logger.debug("position_delta : déplacement géré au niveau pipeline")

        else:
            logger.warning("Paramètre inconnu : '%s'", param)

        return result

    def apply_position_delta(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        dx: float,
        dy: float,
    ) -> tuple:
        """
        Déplace l'objet masqué de (dx, dy) pixels.

        Returns:
            (image_moved, new_mask) — image avec objet déplacé et nouveau masque.
        """
        h, w = image.shape[:2]
        dx_i = int(round(dx))
        dy_i = int(round(dy))

        # Extrait la région de l'objet
        M = np.float32([[1, 0, dx_i], [0, 1, dy_i]])

        # Déplace le masque
        new_mask = cv2.warpAffine(
            mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST
        ).astype(bool)

        # Déplace les pixels de l'objet
        result = image.copy()
        # Efface l'ancienne position (sera rempli par CavityInpainter)
        result[mask.astype(bool)] = 0

        # Copie à la nouvelle position
        shifted_region = cv2.warpAffine(
            image * mask[:, :, np.newaxis].astype(np.uint8), M, (w, h)
        )
        result[new_mask] = shifted_region[new_mask]

        return result, new_mask

    # ------------------------------------------------------------------
    # Corrections individuelles
    # ------------------------------------------------------------------

    def _apply_color_delta(
        self, image: np.ndarray, mask: np.ndarray, delta: float
    ) -> np.ndarray:
        """Shift de luminosité/teinte sur la région masquée."""
        result = image.astype(np.float32)
        shift = delta * 255.0
        result[mask] = np.clip(result[mask] + shift, 0, 255)
        return result.astype(np.uint8)

    def _apply_brightness(
        self, image: np.ndarray, mask: np.ndarray, delta: float
    ) -> np.ndarray:
        """Correction lumière ambiante (multiplicateur)."""
        result = image.astype(np.float32)
        factor = 1.0 + delta
        result[mask] = np.clip(result[mask] * factor, 0, 255)
        return result.astype(np.uint8)

    def _apply_reflectivity(
        self, image: np.ndarray, mask: np.ndarray, delta: float
    ) -> np.ndarray:
        """
        Simule une modification de réflectivité par sur/sous-exposition
        des pixels les plus lumineux.
        """
        result = image.astype(np.float32)
        region = result[mask]
        gray = np.mean(region, axis=-1)
        # Amplifie les highlights proportionnellement
        highlight_mask = gray > (200 if delta > 0 else 128)
        boost = delta * 50.0
        region[highlight_mask] = np.clip(region[highlight_mask] + boost, 0, 255)
        result[mask] = region
        return result.astype(np.uint8)

    def _apply_texture(
        self, image: np.ndarray, mask: np.ndarray, delta: float
    ) -> np.ndarray:
        """
        Modification rugosité : sharpen si delta > 0, blur si delta < 0.
        """
        result = image.copy()
        if delta > 0:
            # Sharpen léger
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            alpha = min(1.0, abs(delta))
            blended = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)
        else:
            # Blur léger
            k = max(1, int(abs(delta) * 10))
            if k % 2 == 0:
                k += 1
            blended = cv2.GaussianBlur(image, (k, k), 0)

        result[mask] = blended[mask]
        return result

    def _apply_shadow(
        self, image: np.ndarray, mask: np.ndarray, delta: float
    ) -> np.ndarray:
        """Assombrit les bords du masque pour simuler une ombre portée."""
        result = image.astype(np.float32)
        # Dilate le masque pour créer une zone d'ombre périmétrique
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
        shadow_zone = (dilated.astype(bool)) & (~mask)
        result[shadow_zone] = np.clip(
            result[shadow_zone] * (1.0 - delta * 0.5), 0, 255
        )
        return result.astype(np.uint8)

    def _apply_refraction(
        self, image: np.ndarray, mask: np.ndarray, delta: float
    ) -> np.ndarray:
        """
        Simule la réfraction (verre) par déformation radiale légère.

        AGENT : uniquement verre_texture.
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Coordonnées du centre du masque
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return result
        cx, cy = xs.mean(), ys.mean()

        # Crée une grille de déformation radiale
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx_grid = (xx - cx) * delta * 0.05
        dy_grid = (yy - cy) * delta * 0.05
        map_x = np.clip((xx + dx_grid), 0, w - 1)
        map_y = np.clip((yy + dy_grid), 0, h - 1)

        distorted = cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        result[mask] = distorted[mask]
        return result

    def _apply_scale(
        self, image: np.ndarray, mask: np.ndarray, delta: float
    ) -> np.ndarray:
        """Redimensionne l'objet masqué (facteur = 1 + delta)."""
        result = image.copy()
        h, w = image.shape[:2]

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return result

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        factor = max(0.1, 1.0 + delta)

        obj_w = x2 - x1 + 1
        obj_h = y2 - y1 + 1
        new_w = max(1, int(obj_w * factor))
        new_h = max(1, int(obj_h * factor))

        crop = image[y1:y2 + 1, x1:x2 + 1]
        scaled = cv2.resize(crop, (new_w, new_h))

        # Centre le résultat sur la bbox originale
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx1 = max(0, cx - new_w // 2)
        ny1 = max(0, cy - new_h // 2)
        nx2 = min(w, nx1 + new_w)
        ny2 = min(h, ny1 + new_h)

        # Efface l'ancienne région (le fond sera restauré par CavityInpainter)
        result[y1:y2 + 1, x1:x2 + 1][mask[y1:y2 + 1, x1:x2 + 1]] = \
            image[y1:y2 + 1, x1:x2 + 1][mask[y1:y2 + 1, x1:x2 + 1]]

        # Copie la version redimensionnée
        sw = nx2 - nx1
        sh = ny2 - ny1
        result[ny1:ny2, nx1:nx2] = scaled[:sh, :sw]

        return result

    def _apply_rotation(
        self, image: np.ndarray, mask: np.ndarray, delta: float
    ) -> np.ndarray:
        """Pivote l'objet masqué de delta degrés."""
        result = image.copy()
        h, w = image.shape[:2]

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return result

        cx, cy = float(xs.mean()), float(ys.mean())
        M = cv2.getRotationMatrix2D((cx, cy), delta, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        new_mask = cv2.warpAffine(
            mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST
        ).astype(bool)

        result[new_mask] = rotated[new_mask]
        return result
