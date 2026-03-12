"""
PhotoEditorPipeline — Orchestrateur principal.

Séquence obligatoire (RÈGLE 1) :
  SAM → MiDaS → Signature → SpatialField → SceneGraph →
  Encode → Route → Modify → Propagate → Inpaint → Store
"""

import numpy as np

from .cavity_inpainter import CavityInpainter
from .hybrid_router import HybridRouter
from .modification_engine import ObjectModificationEngine
from .scene_graph import SceneGraph
from .semantic_encoder import SemanticEncoder
from .spatial_field import SpatialField
from .visual_memory import VisualMemory


class PhotoEditorPipeline:
    """Main orchestrator for the interactive photo editor."""

    def __init__(self, image: np.ndarray, depth_map: np.ndarray):
        h, w = image.shape[:2]
        self.image = image.copy()
        self.depth_map = depth_map.astype(np.float32)

        self.spatial_field = SpatialField(h, w)
        self.scene_graph = SceneGraph()
        self.memory = VisualMemory()
        self.encoder = SemanticEncoder()
        self.router = HybridRouter()
        self.engine = ObjectModificationEngine()
        self.inpainter = CavityInpainter()

        self._objects: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def add_object(
        self,
        object_id: str,
        mask: np.ndarray,
        bbox: tuple,
        crop: np.ndarray,
    ) -> None:
        """Register a segmented object in the pipeline."""
        depth_mean = float(self.depth_map[mask.astype(bool)].mean()) if mask.any() else 0.0

        sig = self.memory.extract_signature(crop, mask, self.depth_map, self.spatial_field)

        self._objects[object_id] = {
            "mask": mask,
            "bbox": bbox,
            "crop": crop,
            "depth_mean": depth_mean,
            "signature": sig,
        }

        self.scene_graph.add_node(
            node_id=object_id,
            mask=mask,
            bbox=bbox,
            depth_mean=depth_mean,
            signature=sig,
        )

    def _rebuild_spatial_field(self) -> None:
        """Rebuild SpatialField from all registered objects."""
        segments = [
            {"mask": obj["mask"], "bbox": obj["bbox"]}
            for obj in self._objects.values()
        ]
        signatures = [obj["signature"] for obj in self._objects.values()]
        self.spatial_field.build_from_scene(self.depth_map, segments, signatures)
        self.scene_graph.build_edges()

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process_modification(
        self,
        image: np.ndarray,
        object_id: str,
        mask: np.ndarray,
        crop: np.ndarray,
        delta_params: dict,
        mask_moved: bool = False,
    ) -> dict:
        """Run the full pipeline for a modification request.

        RÈGLE 1 : Séquence obligatoire respectée.
        RÈGLE 7 : Cavité si mask_moved=True AVANT propagation.

        Returns
        -------
        dict with keys: image, impacts, gain, scene_hash
        """
        result_image = image.copy()

        # 1. Rebuild spatial field with current state
        self._rebuild_spatial_field()

        # 2. Extract signature
        sig = self.memory.extract_signature(crop, mask, self.depth_map, self.spatial_field)
        scene_hash = self.spatial_field.compute_scene_hash()

        # 3. Find similar corrections (with scene_hash filter — RÈGLE 3)
        similar = self.memory.find_similar(sig, scene_hash)

        # 4. Get scene relations (SceneGraph light — topology only)
        relations = self.scene_graph.get_relations(object_id)

        # 5. Encode metrics to text (RÈGLE 2 — never raw numbers to LLM)
        prompt, confidence = self.encoder.encode(
            signature=sig,
            delta_metrics=delta_params,
            similar_corrections=similar,
            relations=relations,
        )

        # 6. Route to LLM (RÈGLE 5 — Qwen 14B Q4 by default)
        mat = sig.get("material_class", "surface_generique")
        error_history = self.memory.get_history(object_id)
        correction, model_used = self.router.route(
            prompt=prompt,
            confidence=confidence,
            material_class=mat,
            error_count=len(error_history),
        )

        # If LLM returned no correction, use delta_params directly
        if correction is None:
            correction = delta_params

        # 7. RÈGLE 7 — Cavity inpainting BEFORE propagation if object moved
        if mask_moved:
            result_image = self.inpainter.inpaint(result_image, mask)

        # 8. Apply modification (RÈGLE 4 — material check inside engine)
        result_image = self.engine.apply(result_image, mask, correction, mat)

        # 9. RÈGLE 1 — Propagate modification in SpatialField (NEVER skip)
        self.spatial_field.propagate_modification(
            mask.astype(bool), delta_params
        )

        # 10. Get impacted objects (SceneGraph topology)
        impacts = self.scene_graph.propagate_changes(object_id)

        # 11. Compute quality gain (simple metric)
        score_before = float(np.mean(image[mask.astype(bool)]) / 255.0)
        score_after = float(np.mean(result_image[mask.astype(bool)]) / 255.0)
        gain = score_after - score_before

        # 12. Store correction in memory (RÈGLE 10 — with scene_hash)
        new_scene_hash = self.spatial_field.compute_scene_hash()
        self.memory.store(
            object_id=object_id,
            crop=crop,
            mask=mask,
            correction=correction,
            model=model_used,
            score_before=score_before,
            score_after=score_after,
            scene_hash=new_scene_hash,
            signature=sig,
        )

        self.image = result_image
        return {
            "image": result_image,
            "impacts": impacts,
            "gain": gain,
            "scene_hash": new_scene_hash,
        }
