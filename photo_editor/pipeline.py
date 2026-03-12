"""
pipeline.py — PhotoEditorPipeline
Orchestrateur principal du pipeline Photo Editor Interactif.

Séquence obligatoire :
  SAM → MiDaS → Signature → SpatialField → SceneGraph →
  Encode → Route → Modify → Propagate → Inpaint → Store

RÈGLE 1 — SÉQUENCE PIPELINE :
  Ne jamais sauter SpatialField.propagate_modification().

RÈGLE 7 — CAVITÉ OBLIGATOIRE :
  Si mask_moved=True → CavityInpainter AVANT propagation lumière.
"""

import logging
from typing import Optional

import numpy as np

from .cavity_inpainter import CavityInpainter
from .hybrid_router import HybridRouter
from .modification_engine import ObjectModificationEngine
from .scene_graph import SceneGraph
from .semantic_encoder import SemanticEncoder
from .spatial_field import SpatialField
from .visual_memory import VisualMemory

logger = logging.getLogger(__name__)


class PhotoEditorPipeline:
    """
    Orchestrateur complet du pipeline Photo Editor.

    Usage :
        pipeline = PhotoEditorPipeline(image, depth_map)
        pipeline.add_object("verre_01", mask, bbox, crop)
        result = pipeline.process_modification(
            image, "verre_01", mask, crop,
            delta_params={"color_delta": 0.1},
            mask_moved=False
        )
    """

    def __init__(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        memory_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            image      : image source (H, W, 3) uint8.
            depth_map  : carte profondeur float32 (H, W) issue de MiDaS small.
            memory_path: chemin optionnel pour visual_memory.json.
        """
        h, w = image.shape[:2]
        self.image = image
        self.depth_map = depth_map

        # Modules du pipeline
        self.spatial_field = SpatialField(h, w)
        self.scene_graph = SceneGraph()
        self.visual_memory = VisualMemory(memory_path)
        self.semantic_encoder = SemanticEncoder()
        self.hybrid_router = HybridRouter()
        self.modification_engine = ObjectModificationEngine()
        self.cavity_inpainter = CavityInpainter()

        # Objets enregistrés {object_id: {mask, bbox, crop, signature}}
        self._objects: dict = {}

        # Initialise le SpatialField avec les données initiales
        self.spatial_field.build_from_scene(depth_map, [], [])

    # ------------------------------------------------------------------
    # Gestion des objets
    # ------------------------------------------------------------------

    def add_object(
        self,
        object_id: str,
        mask: np.ndarray,
        bbox: tuple,
        crop: np.ndarray,
    ) -> dict:
        """
        Enregistre un objet segmenté (depuis SAM) dans le pipeline.

        Args:
            object_id : identifiant unique.
            mask      : masque binaire (H, W).
            bbox      : (x1, y1, x2, y2).
            crop      : région recadrée (H_obj, W_obj, 3).

        Returns:
            Signature 8D calculée.
        """
        # Calcule la signature
        sig = self.visual_memory.extract_signature(
            crop, mask, self.depth_map, self.spatial_field
        )

        depth_mean = float(sig.get("depth_mean", 0.5))
        material = sig.get("material_class", "surface_generique")
        logger.info(
            "Objet ajouté : %s | matériau=%s | depth=%.3f",
            object_id, material, depth_mean,
        )

        self._objects[object_id] = {
            "mask": mask,
            "bbox": bbox,
            "crop": crop,
            "signature": sig,
        }

        # Ajoute au SceneGraph
        self.scene_graph.add_node(
            object_id, mask, bbox, depth_mean, signature=sig
        )

        # Met à jour le SpatialField
        self.spatial_field.build_from_scene(
            self.depth_map,
            [obj["mask"] for obj in self._objects.values()],
            [obj["signature"] for obj in self._objects.values()],
        )

        return sig

    # ------------------------------------------------------------------
    # Modification principale
    # ------------------------------------------------------------------

    def process_modification(
        self,
        image: np.ndarray,
        object_id: str,
        mask: np.ndarray,
        crop: np.ndarray,
        delta_params: dict,
        mask_moved: bool = False,
        sigma: float = 15.0,
    ) -> dict:
        """
        Applique une modification complète (séquence pipeline obligatoire).

        Séquence :
          Signature → SpatialField → SceneGraph → Encode → Route →
          Modify → [Inpaint si mask_moved] → Propagate → Store

        Args:
            image        : image courante (H, W, 3) uint8.
            object_id    : identifiant de l'objet.
            mask         : masque binaire (H, W).
            crop         : région recadrée de l'objet.
            delta_params : paramètres initiaux de modification.
            mask_moved   : True si l'objet est déplacé (active CavityInpainter).
            sigma        : rayon propagation SpatialField (défaut 15px).

        Returns:
            dict {
                "image"      : image finale (H, W, 3) uint8,
                "impacts"    : liste d'objets voisins impactés,
                "gain"       : amélioration qualité (float),
                "scene_hash" : hash état scène post-modification,
                "correction" : dict correction appliquée,
            }
        """
        # --- Étape 1 : Signature + état scène ---
        sig = self.visual_memory.extract_signature(
            crop, mask, self.depth_map, self.spatial_field
        )
        scene_hash_before = self.spatial_field.compute_scene_hash()
        material_class = sig.get("material_class", "surface_generique")

        # --- Étape 2 : Recherche corrections similaires (avec scene_hash) ---
        similar = self.visual_memory.find_similar(
            sig, scene_hash_before, threshold=0.80
        )

        # --- Étape 3 : Métriques pipeline ---
        pipeline_metrics = self._compute_pipeline_metrics(image, mask, crop)

        # --- Étape 4 : Encodage sémantique ---
        spatial_props = None
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            spatial_props = self.spatial_field.sample(cx, cy)

        scene_edges = self.scene_graph.get_edges(object_id)

        prompt, confidence = self.semantic_encoder.encode(
            object_id=object_id,
            signature=sig,
            spatial_props=spatial_props,
            scene_relations=scene_edges,
            similar_corrections=similar,
            pipeline_metrics=pipeline_metrics,
        )

        logger.info(
            "Objet %s | confiance=%.2f | matériau=%s | cloud=%s",
            object_id, confidence, material_class,
            confidence < 0.7 or material_class in ("fourrure_textile", "verre_texture"),
        )

        # --- Étape 5 : Routage LLM ---
        correction = self.hybrid_router.route(prompt, confidence, material_class)
        logger.info("Correction LLM : %s", correction)

        # --- Étape 6 : Application modification ---
        result_image = self.modification_engine.apply(
            image, mask, correction, material_class=material_class
        )

        # Si déplacement → appliquer position_delta
        if delta_params.get("position_delta") and isinstance(
            delta_params["position_delta"], (list, tuple)
        ):
            dx, dy = delta_params["position_delta"]
            result_image, mask = self.modification_engine.apply_position_delta(
                result_image, mask, dx, dy
            )
            mask_moved = True

        # --- Étape 7 : CavityInpainter si objet déplacé ---
        # RÈGLE 7 : CavityInpainter AVANT propagation lumière
        if mask_moved:
            logger.info("mask_moved=True → CavityInpainter activé")
            original_mask = self._objects.get(object_id, {}).get("mask", mask)
            result_image = self.cavity_inpainter.fill(result_image, original_mask)

        # --- Étape 8 : Propagation SpatialField ---
        # RÈGLE 1 : TOUJOURS appeler après ObjectModificationEngine
        self.spatial_field.propagate_modification(
            mask, correction, sigma=sigma
        )

        # --- Étape 9 : Impacts topologiques ---
        impacted_nodes = self.scene_graph.propagate_changes(object_id)
        impacts = [n.id for n in impacted_nodes]

        # --- Étape 10 : Score qualité ---
        score_before = pipeline_metrics.get("quality_score", 0.5)
        score_after = self._compute_quality_score(result_image, mask)
        gain = score_after - score_before

        # --- Étape 11 : Stockage en mémoire ---
        scene_hash_after = self.spatial_field.compute_scene_hash()
        self.visual_memory.store(
            object_id=object_id,
            crop=crop,
            mask=mask,
            correction=correction,
            model=correction.get("_model", "unknown"),
            score_before=score_before,
            score_after=score_after,
            scene_hash=scene_hash_after,
        )

        # Met à jour l'objet enregistré
        if object_id in self._objects:
            self._objects[object_id]["mask"] = mask
            self._objects[object_id]["signature"] = sig

        if gain > 0:
            self.semantic_encoder.reset_error_count(object_id)

        return {
            "image": result_image,
            "impacts": impacts,
            "gain": gain,
            "scene_hash": scene_hash_after,
            "correction": correction,
        }

    # ------------------------------------------------------------------
    # Métriques qualité
    # ------------------------------------------------------------------

    def _compute_pipeline_metrics(
        self, image: np.ndarray, mask: np.ndarray, crop: np.ndarray
    ) -> dict:
        """
        Calcule les métriques brutes de qualité de fusion.

        Returns:
            dict {"gradient_error", "ssim", "quality_score"}
        """
        try:
            from skimage.metrics import structural_similarity as ssim_fn  # type: ignore
        except ImportError:
            ssim_fn = None

        h, w = image.shape[:2]
        mask_bool = mask.astype(bool)

        # Gradient error : discontinuité de gradient à la frontière du masque
        gray = np.mean(image, axis=2).astype(np.float32) / 255.0
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Zone frontière (dilatation - masque)
        import cv2
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel)
        border = (dilated.astype(bool)) & (~mask_bool)

        if border.any():
            inner_grad = float(np.mean(grad_mag[mask_bool])) if mask_bool.any() else 0.0
            outer_grad = float(np.mean(grad_mag[border]))
            gradient_error = abs(inner_grad - outer_grad)
        else:
            gradient_error = 0.0

        # SSIM simplifié
        ssim_val = 1.0
        if ssim_fn is not None and mask_bool.any():
            try:
                ys, xs = np.where(mask_bool)
                y1, y2 = max(0, ys.min() - 5), min(h, ys.max() + 5)
                x1, x2 = max(0, xs.min() - 5), min(w, xs.max() + 5)
                region = gray[y1:y2, x1:x2]
                if region.size > 0:
                    ssim_val = float(ssim_fn(region, region))
            except Exception:
                ssim_val = 1.0

        quality_score = (1.0 - min(1.0, gradient_error)) * 0.6 + ssim_val * 0.4

        return {
            "gradient_error": gradient_error,
            "ssim": ssim_val,
            "quality_score": quality_score,
        }

    def _compute_quality_score(
        self, image: np.ndarray, mask: np.ndarray
    ) -> float:
        """Score de qualité simplifié post-modification."""
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return 1.0
        gray = np.mean(image, axis=2).astype(np.float32) / 255.0
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        inner_grad = float(np.mean(grad_mag[mask_bool]))
        return float(np.clip(1.0 - inner_grad, 0.0, 1.0))
