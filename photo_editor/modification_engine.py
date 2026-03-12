"""
ObjectModificationEngine — Apply corrections from LLM onto object masks.

AGENT : Toujours vérifier material_class avant d'appliquer refraction
        (uniquement verre_texture) ou reflectivity_delta (metal + verre).
"""

import cv2
import numpy as np


class ObjectModificationEngine:
    """Apply one LLM correction on the masked region of an image."""

    # Materials that support reflectivity changes
    _REFLECTIVE_MATERIALS = {"metal_speculaire", "verre_texture"}

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        correction: dict,
        material_class: str = "surface_generique",
    ) -> np.ndarray:
        """Apply a single correction dict on the image region defined by *mask*.

        Parameters
        ----------
        image           : (H, W, 3) BGR image.
        mask            : (H, W) bool or uint8 mask.
        correction      : dict with keys like position_delta, color_delta, etc.
        material_class  : Material of the object for safety checks.

        Returns
        -------
        Modified image (copy).
        """
        result = image.copy()
        bool_mask = mask.astype(bool)

        # -- Position delta (translate object) --
        if "position_delta" in correction:
            dx, dy = 0, 0
            pd = correction["position_delta"]
            if isinstance(pd, (list, tuple)) and len(pd) == 2:
                dx, dy = int(pd[0]), int(pd[1])
            elif isinstance(pd, (int, float)):
                dx, dy = int(pd), 0
            result = self._translate_object(result, bool_mask, dx, dy)

        # -- Scale delta --
        if "scale_delta" in correction:
            scale = float(correction["scale_delta"])
            result = self._scale_object(result, bool_mask, scale)

        # -- Rotation delta --
        if "rotation_delta" in correction:
            angle = float(correction["rotation_delta"])
            result = self._rotate_object(result, bool_mask, angle)

        # -- Color delta (brightness shift) --
        if "color_delta" in correction:
            delta = float(correction["color_delta"])
            result = self._adjust_brightness(result, bool_mask, delta)

        # -- Texture delta (roughness) --
        if "texture_delta" in correction:
            delta = float(correction["texture_delta"])
            result = self._adjust_texture(result, bool_mask, delta)

        # -- Reflectivity delta (metal + verre only) --
        if "reflectivity_delta" in correction:
            # AGENT : protection matériau
            if material_class in self._REFLECTIVE_MATERIALS:
                delta = float(correction["reflectivity_delta"])
                result = self._adjust_reflectivity(result, bool_mask, delta)

        # -- Lumiere ambiante --
        if "lumiere_ambiante" in correction:
            delta = float(correction["lumiere_ambiante"])
            result = self._adjust_brightness(result, bool_mask, delta * 0.5)

        # -- Shadow intensity --
        if "shadow_intensity" in correction:
            delta = float(correction["shadow_intensity"])
            result = self._adjust_brightness(result, bool_mask, -delta * 0.3)

        # -- Refraction (verre_texture only) --
        if "refraction" in correction:
            # AGENT : uniquement verre_texture
            if material_class == "verre_texture":
                idx = float(correction["refraction"])
                result = self._apply_refraction(result, bool_mask, idx)

        return result

    # ------------------------------------------------------------------
    # Internal transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_object(image: np.ndarray, mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
        h, w = image.shape[:2]
        obj_pixels = image.copy()
        obj_pixels[~mask] = 0
        translated = np.zeros_like(image)

        src_y, src_x = np.where(mask)
        dst_y, dst_x = src_y + dy, src_x + dx
        valid = (dst_y >= 0) & (dst_y < h) & (dst_x >= 0) & (dst_x < w)
        translated[dst_y[valid], dst_x[valid]] = obj_pixels[src_y[valid], src_x[valid]]

        result = image.copy()
        result[mask] = 0  # clear original location
        non_zero = translated.any(axis=2) if translated.ndim == 3 else translated > 0
        result[non_zero] = translated[non_zero]
        return result

    @staticmethod
    def _scale_object(image: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale) < 1e-6:
            return image
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return image
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        crop = image[y0:y1 + 1, x0:x1 + 1].copy()
        crop_mask = mask[y0:y1 + 1, x0:x1 + 1]

        new_h = max(1, int(crop.shape[0] * (1 + scale)))
        new_w = max(1, int(crop.shape[1] * (1 + scale)))
        resized = cv2.resize(crop, (new_w, new_h))
        resized_mask = cv2.resize(crop_mask.astype(np.uint8), (new_w, new_h)) > 0

        result = image.copy()
        result[mask] = 0

        # Center the resized crop
        ny0 = max(0, cy - new_h // 2)
        nx0 = max(0, cx - new_w // 2)
        ny1 = min(image.shape[0], ny0 + new_h)
        nx1 = min(image.shape[1], nx0 + new_w)
        rh, rw = ny1 - ny0, nx1 - nx0
        sub_resized = resized[:rh, :rw]
        sub_mask = resized_mask[:rh, :rw]
        result[ny0:ny1, nx0:nx1][sub_mask] = sub_resized[sub_mask]
        return result

    @staticmethod
    def _rotate_object(image: np.ndarray, mask: np.ndarray, angle: float) -> np.ndarray:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return image
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        pad = 10
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(image.shape[0], y1 + pad + 1)
        x1 = min(image.shape[1], x1 + pad + 1)

        crop = image[y0:y1, x0:x1].copy()
        ch, cw = crop.shape[:2]
        center = (cx - x0, cy - y0)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(crop, mat, (cw, ch))

        crop_mask = mask[y0:y1, x0:x1].astype(np.uint8)
        rot_mask = cv2.warpAffine(crop_mask, mat, (cw, ch)) > 0

        result = image.copy()
        result[y0:y1, x0:x1][rot_mask] = rotated[rot_mask]
        return result

    @staticmethod
    def _adjust_brightness(image: np.ndarray, mask: np.ndarray, delta: float) -> np.ndarray:
        result = image.copy()
        region = result[mask].astype(np.float32)
        region += delta * 255
        result[mask] = np.clip(region, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def _adjust_texture(image: np.ndarray, mask: np.ndarray, delta: float) -> np.ndarray:
        """Simulate roughness change via controlled blur/sharpen."""
        result = image.copy()
        if delta > 0:
            # Rougher → slight blur
            ksize = max(3, int(delta * 10) | 1)
            blurred = cv2.GaussianBlur(result, (ksize, ksize), 0)
            result[mask] = blurred[mask]
        elif delta < 0:
            # Smoother → sharpen
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            sharpened = cv2.filter2D(result, -1, kernel)
            result[mask] = sharpened[mask]
        return result

    @staticmethod
    def _adjust_reflectivity(image: np.ndarray, mask: np.ndarray, delta: float) -> np.ndarray:
        result = image.copy()
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[mask, 2] += delta * 50  # increase value channel
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result

    @staticmethod
    def _apply_refraction(image: np.ndarray, mask: np.ndarray, index: float) -> np.ndarray:
        """Simulate refraction distortion on glass-like materials."""
        result = image.copy()
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return result
        shift = int(index * 3)
        shifted_x = np.clip(xs + shift, 0, image.shape[1] - 1)
        result[ys, xs] = image[ys, shifted_x]
        return result
