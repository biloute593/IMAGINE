"""
semantic_encoder.py — SemanticEncoder métriques → texte pour LLM
Traduit les métriques numériques en rapport textuel compréhensible.

RÈGLE 2 — SÉMANTISATION OBLIGATOIRE :
  Toute métrique numérique DOIT passer ici avant d'être envoyée au LLM.
  Jamais de chiffres bruts dans les prompts.
"""

from __future__ import annotations

from typing import Optional


# Seuils de confiance
CONFIDENCE_NORMAL = 0.85
CONFIDENCE_HIGH_ERROR = 0.50
CONFIDENCE_RECURRENT = 0.30


class SemanticEncoder:
    """
    Convertit les métriques numériques du pipeline en rapport textuel
    structuré pour le LLM hybride.
    """

    def __init__(self) -> None:
        self._error_history: dict[str, int] = {}  # object_id → nb erreurs récurrentes

    # ------------------------------------------------------------------
    # Encodage principal
    # ------------------------------------------------------------------

    def encode(
        self,
        object_id: str,
        signature: dict,
        spatial_props: Optional[dict] = None,
        scene_relations: Optional[list] = None,
        similar_corrections: Optional[list] = None,
        pipeline_metrics: Optional[dict] = None,
    ) -> tuple:
        """
        Produit le prompt textuel et le score de confiance.

        Args:
            object_id         : identifiant de l'objet.
            signature         : signature 8D de l'objet.
            spatial_props     : propriétés SpatialField (sample()).
            scene_relations   : liste d'edges du SceneGraph.
            similar_corrections: corrections similaires trouvées en mémoire.
            pipeline_metrics  : métriques brutes (gradient_error, ssim, etc.).

        Returns:
            (prompt_str, confidence_float)
        """
        parts = []

        # --- Matériau et propriétés physiques ---
        mat = signature.get("material_class", "surface_generique")
        brightness = signature.get("brightness", 0.5)
        specularity = signature.get("specularity", 0.0)
        roughness = signature.get("roughness", 0.0)
        depth_mean = signature.get("depth_mean", 0.5)
        texture_entropy = signature.get("texture_entropy", 4.0)

        mat_desc = self._describe_material(mat, specularity, roughness)
        parts.append(f"Matériau détecté : {mat_desc}.")

        bright_desc = self._describe_brightness(brightness)
        parts.append(f"Luminosité : {bright_desc}.")

        depth_desc = self._describe_depth(depth_mean)
        parts.append(f"Position dans la scène : {depth_desc}.")

        entropy_desc = self._describe_entropy(texture_entropy)
        parts.append(f"Complexité de texture : {entropy_desc}.")

        # --- Propriétés SpatialField ---
        if spatial_props:
            refl = spatial_props.get("reflectivity", 0.0)
            lum = spatial_props.get("luminance", 0.5)
            refl_desc = self._describe_reflectivity(refl)
            lum_diff = lum - brightness
            lum_delta_desc = self._describe_luminance_delta(lum_diff)
            parts.append(f"Réflectivité de surface : {refl_desc}.")
            parts.append(f"Écart luminosité vs fond : {lum_delta_desc}.")

        # --- Relations de scène ---
        if scene_relations:
            rel_desc = self._describe_relations(scene_relations)
            if rel_desc:
                parts.append(f"Relations avec autres objets : {rel_desc}.")

        # --- Métriques pipeline ---
        gradient_error = 0.0
        if pipeline_metrics:
            gradient_error = pipeline_metrics.get("gradient_error", 0.0)
            ssim = pipeline_metrics.get("ssim", 1.0)
            metrics_desc = self._describe_metrics(gradient_error, ssim)
            parts.append(f"Métriques pipeline : {metrics_desc}.")

        # --- Corrections similaires en mémoire ---
        if similar_corrections:
            mem_desc = self._describe_memory(similar_corrections)
            parts.append(f"Mémoire corrections similaires : {mem_desc}.")

        # --- Instruction format réponse ---
        parts.append(
            "\nRéponds UNIQUEMENT en JSON strict : "
            '{"param": "nom_parametre", "delta": 0.0, "raison": "explication courte"}'
        )

        prompt = "\n".join(parts)

        # --- Calcul confiance ---
        confidence = self._compute_confidence(
            object_id, gradient_error, pipeline_metrics
        )

        return prompt, confidence

    # ------------------------------------------------------------------
    # Calcul confiance
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        object_id: str,
        gradient_error: float,
        pipeline_metrics: Optional[dict],
    ) -> float:
        # Vérifie les erreurs récurrentes
        recurrence = self._error_history.get(object_id, 0)
        if recurrence >= 2:
            return CONFIDENCE_RECURRENT

        if gradient_error > 0.4:
            self._error_history[object_id] = recurrence + 1
            return CONFIDENCE_HIGH_ERROR

        if pipeline_metrics:
            ssim = pipeline_metrics.get("ssim", 1.0)
            if ssim < 0.5:
                return CONFIDENCE_HIGH_ERROR

        return CONFIDENCE_NORMAL

    def reset_error_count(self, object_id: str) -> None:
        """Réinitialise le compteur d'erreurs récurrentes pour un objet."""
        self._error_history.pop(object_id, None)

    # ------------------------------------------------------------------
    # Descriptions textuelles
    # ------------------------------------------------------------------

    def _describe_material(
        self, mat: str, specularity: float, roughness: float
    ) -> str:
        descs = {
            "metal_speculaire": f"métal spéculaire (reflets intenses, spécularité={specularity:.0%})",
            "verre_texture": f"verre ou surface transparente (spécularité={specularity:.0%})",
            "fourrure_textile": f"textile ou fourrure (texture rugueuse, roughness={roughness:.3f})",
            "plastique_mat": "plastique mat (surface lisse, faible réflexion)",
            "surface_generique": "surface générique",
        }
        return descs.get(mat, "matériau inconnu")

    def _describe_brightness(self, brightness: float) -> str:
        if brightness > 0.8:
            return "très lumineuse (surexposition probable)"
        if brightness > 0.6:
            return "lumineuse"
        if brightness > 0.4:
            return "luminosité normale"
        if brightness > 0.2:
            return "sombre"
        return "très sombre (sous-exposition probable)"

    def _describe_depth(self, depth_mean: float) -> str:
        if depth_mean < 0.2:
            return "très proche du premier plan"
        if depth_mean < 0.4:
            return "premier plan"
        if depth_mean < 0.6:
            return "plan médian"
        if depth_mean < 0.8:
            return "arrière-plan"
        return "très loin en arrière-plan"

    def _describe_entropy(self, entropy: float) -> str:
        if entropy > 6.0:
            return "très complexe (détails fins nombreux)"
        if entropy > 4.0:
            return "texture modérément complexe"
        if entropy > 2.0:
            return "texture simple"
        return "texture très uniforme (peu de détails)"

    def _describe_reflectivity(self, refl: float) -> str:
        if refl > 0.7:
            return "très réfléchissante (proche miroir)"
        if refl > 0.4:
            return "réflectivité élevée"
        if refl > 0.15:
            return "réflectivité modérée"
        return "surface mate (très faible réflexion)"

    def _describe_luminance_delta(self, delta: float) -> str:
        if delta > 0.3:
            return "objet nettement plus lumineux que son environnement"
        if delta > 0.1:
            return "objet légèrement surexposé par rapport au fond"
        if delta < -0.3:
            return "objet nettement plus sombre que son environnement"
        if delta < -0.1:
            return "objet légèrement sous-exposé par rapport au fond"
        return "cohérence luminosité objet/fond correcte"

    def _describe_metrics(self, gradient_error: float, ssim: float) -> str:
        parts = []
        if gradient_error > 0.4:
            parts.append(
                f"discontinuité de gradient sévère aux bords de l'objet "
                f"(correction urgente requise)"
            )
        elif gradient_error > 0.2:
            parts.append("légère discontinuité de gradient aux bords")
        else:
            parts.append("continuité de gradient correcte")

        if ssim < 0.5:
            parts.append("similarité structurelle très dégradée")
        elif ssim < 0.8:
            parts.append("similarité structurelle acceptable")
        else:
            parts.append("bonne cohérence structurelle")

        return " ; ".join(parts)

    def _describe_relations(self, edges: list) -> str:
        if not edges:
            return ""
        descs = []
        rel_map = {
            "occludes": "occulte",
            "touches": "touche",
            "adjacent": "est adjacent à",
            "supports": "est supporté par",
        }
        for edge in edges[:3]:  # max 3 relations dans le prompt
            src = edge.source if hasattr(edge, "source") else edge.get("source", "?")
            tgt = edge.target if hasattr(edge, "target") else edge.get("target", "?")
            rel = edge.relation if hasattr(edge, "relation") else edge.get("relation", "?")
            verb = rel_map.get(rel, rel)
            descs.append(f"'{src}' {verb} '{tgt}'")
        return ", ".join(descs)

    def _describe_memory(self, similar: list) -> str:
        if not similar:
            return "aucune correction similaire trouvée"
        best = similar[0]
        score = best.get("similarity", 0.0)
        rec = best.get("record", {})
        corr = rec.get("correction", {})
        gain = rec.get("gain", 0.0)
        model = rec.get("model", "inconnu")
        return (
            f"{len(similar)} correction(s) similaire(s) trouvée(s) "
            f"(meilleure similarité {score:.0%}, "
            f"gain qualité {gain:+.2f}, modèle {model}, "
            f"paramètre appliqué : {corr.get('param', '?')} "
            f"delta={corr.get('delta', 0.0):+.3f})"
        )
