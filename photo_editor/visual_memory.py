"""
visual_memory.py — VisualMemory 8D + scene_hash
Mémoire persistante des corrections avec validation par état de scène.

RÈGLE CRITIQUE — DÉSYNCHRONISATION :
  find_similar() filtre par scene_hash compatible.
  Une correction stockée pour scène A est invalide pour scène B.
  Ne jamais désactiver ce filtre.

  AGENT : C'est le bug le plus courant sur ce projet.
"""

import json
import math
import os
from typing import Optional

import numpy as np

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "visual_memory.json")

# Classes matériau possibles
MATERIAL_CLASSES = [
    "metal_speculaire",
    "verre_texture",
    "fourrure_textile",
    "plastique_mat",
    "surface_generique",
]


class VisualMemory:
    """
    Stocke et retrouve les corrections passées en comparant les signatures
    8D des objets, avec validation obligatoire du scene_hash.

    Signature 8D :
        brightness      — luminosité moyenne objet
        saturation      — écart-type couleur
        specularity     — % pixels très lumineux (reflet)
        roughness       — gradient texture moyen
        depth_mean      — profondeur moyenne (MiDaS)
        material_class  — catégorie matériau
        texture_entropy — complexité texture (entropie Shannon)
        scene_hash      — hash état SpatialField au moment du stockage
    """

    def __init__(self, memory_path: Optional[str] = None) -> None:
        self.memory_path = memory_path or MEMORY_FILE
        self._records: list = []
        self._load()

    # ------------------------------------------------------------------
    # Extraction de signature
    # ------------------------------------------------------------------

    def extract_signature(
        self,
        crop: np.ndarray,
        mask: np.ndarray,
        depth_map: np.ndarray,
        spatial_field,
    ) -> dict:
        """
        Calcule la signature 8D d'un objet.

        Args:
            crop          : image recadrée de l'objet (H, W, 3) uint8.
            mask          : masque binaire (H, W) bool.
            depth_map     : carte profondeur float32 (H_full, W_full).
            spatial_field : instance SpatialField courante.

        Returns:
            dict avec les 8 champs de signature.
        """
        if crop.ndim == 2:
            gray = crop.astype(np.float32) / 255.0
        else:
            gray = np.mean(crop, axis=2).astype(np.float32) / 255.0

        # Pixels valides dans le masque (recadré)
        h_crop, w_crop = gray.shape[:2]
        if mask.shape != gray.shape[:2]:
            m = np.ones((h_crop, w_crop), dtype=bool)
        else:
            m = mask.astype(bool)

        pixels_gray = gray[m] if m.any() else gray.ravel()

        # brightness : luminosité moyenne
        brightness = float(np.mean(pixels_gray))

        # saturation : écart-type sur les canaux couleur
        if crop.ndim == 3:
            sat_pixels = crop[m] if m.any() else crop.reshape(-1, crop.shape[-1])
            saturation = float(np.std(sat_pixels.astype(np.float32) / 255.0))
        else:
            saturation = 0.0

        # specularity : fraction de pixels très lumineux (>0.85)
        specularity = float(np.mean(pixels_gray > 0.85))

        # roughness : gradient moyen (texture)
        if pixels_gray.size > 1:
            gy, gx = np.gradient(gray)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)
            roughness = float(np.mean(grad_mag[m]) if m.any() else np.mean(grad_mag))
        else:
            roughness = 0.0

        # depth_mean : profondeur moyenne dans le masque
        if depth_map is not None and mask.shape == depth_map.shape[:2]:
            dm = depth_map[mask] if mask.any() else depth_map.ravel()
            depth_mean = float(np.mean(dm))
        else:
            depth_mean = 0.5

        # material_class : déterminé heuristiquement
        material_class = self._classify_material(
            brightness, saturation, specularity, roughness
        )

        # texture_entropy : entropie Shannon sur les niveaux de gris
        texture_entropy = self.compute_texture_entropy(crop, m)

        # scene_hash : depuis le SpatialField
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

    def compute_texture_entropy(
        self, crop: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Calcule l'entropie Shannon sur les niveaux de gris du crop.

        Args:
            crop : image (H, W) ou (H, W, 3) uint8.
            mask : masque optionnel (H, W) bool.

        Returns:
            Entropie en bits (float).
        """
        if crop.ndim == 3:
            gray = np.mean(crop, axis=2).astype(np.uint8)
        else:
            gray = crop.astype(np.uint8)

        if mask is not None and mask.shape == gray.shape and mask.any():
            pixels = gray[mask]
        else:
            pixels = gray.ravel()

        if pixels.size == 0:
            return 0.0

        counts = np.bincount(pixels, minlength=256).astype(np.float64)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        return entropy

    # ------------------------------------------------------------------
    # Recherche de corrections similaires
    # ------------------------------------------------------------------

    def find_similar(
        self,
        signature: dict,
        scene_hash: str,
        threshold: float = 0.80,
    ) -> list:
        """
        Trouve les corrections similaires, en filtrant obligatoirement
        par scene_hash compatible.

        RÈGLE CRITIQUE : Ne jamais désactiver le filtre scene_hash.
        Si scene_hash différent → ignorer même si similarité > 0.85.

        Args:
            signature  : signature 8D de l'objet courant.
            scene_hash : hash de l'état courant du SpatialField.
            threshold  : seuil de similarité [0, 1].

        Returns:
            Liste de corrections compatibles, triée par score décroissant.
        """
        results = []
        for record in self._records:
            stored_sig = record.get("signature", {})

            # RÈGLE CRITIQUE — DÉSYNCHRONISATION
            stored_hash = stored_sig.get("scene_hash", "")
            if stored_hash != scene_hash:
                continue

            score = self._similarity(signature, stored_sig)
            if score >= threshold:
                results.append({"record": record, "similarity": score})

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Stockage
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
    ) -> None:
        """
        Sauvegarde une correction avec son contexte de scène.

        Args:
            object_id   : identifiant de l'objet.
            crop        : image recadrée (pour signature).
            mask        : masque binaire.
            correction  : dict correction du LLM.
            model       : nom du modèle utilisé.
            score_before: score qualité avant correction.
            score_after : score qualité après correction.
            scene_hash  : hash état scène au moment de l'application.
        """
        record = {
            "object_id": object_id,
            "signature": {
                "scene_hash": scene_hash,
                "brightness": float(np.mean(crop.astype(np.float32) / 255.0))
                if crop is not None
                else 0.0,
            },
            "correction": correction,
            "model": model,
            "score_before": score_before,
            "score_after": score_after,
            "gain": score_after - score_before,
        }
        self._records.append(record)
        self._save()

    def get_history(self, object_id: str, n: int = 5) -> list:
        """
        Retourne les n dernières corrections pour un objet donné.

        Args:
            object_id : identifiant de l'objet.
            n         : nombre max de résultats.

        Returns:
            Liste des enregistrements les plus récents.
        """
        history = [r for r in self._records if r.get("object_id") == object_id]
        return history[-n:]

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _classify_material(
        self,
        brightness: float,
        saturation: float,
        specularity: float,
        roughness: float,
    ) -> str:
        if specularity > 0.3 and roughness < 0.05:
            return "metal_speculaire"
        if specularity > 0.2 and saturation < 0.1:
            return "verre_texture"
        if roughness > 0.15 and saturation > 0.05:
            return "fourrure_textile"
        if specularity < 0.05 and roughness < 0.08:
            return "plastique_mat"
        return "surface_generique"

    def _similarity(self, sig_a: dict, sig_b: dict) -> float:
        """
        Calcule un score de similarité [0, 1] entre deux signatures 8D.
        Les champs numériques utilisent une distance normalisée.
        Le material_class utilise une comparaison exacte (poids fort).
        """
        numeric_keys = [
            "brightness",
            "saturation",
            "specularity",
            "roughness",
            "depth_mean",
            "texture_entropy",
        ]
        diffs = []
        for key in numeric_keys:
            a = float(sig_a.get(key, 0.0))
            b = float(sig_b.get(key, 0.0))
            # Normalise les entropies (max ~8 bits)
            if key == "texture_entropy":
                a /= 8.0
                b /= 8.0
            diffs.append(abs(a - b))

        numeric_score = 1.0 - min(1.0, float(np.mean(diffs)))

        # Bonus si même matériau
        mat_bonus = 0.2 if sig_a.get("material_class") == sig_b.get("material_class") else 0.0

        return min(1.0, numeric_score * 0.8 + mat_bonus)

    def _load(self) -> None:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    self._records = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._records = []
        else:
            self._records = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)
