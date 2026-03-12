"""
SemanticEncoder — Traduit métriques numériques en rapport textuel pour LLM.
JAMAIS envoyer de chiffres bruts dans les prompts.
"""


class SemanticEncoder:
    """Translate numeric metrics into a natural-language report for the LLM."""

    # AGENT : confidence_score thresholds
    _HIGH_CONF = 0.85
    _MED_CONF = 0.50
    _LOW_CONF = 0.30

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_brightness(delta: float) -> str:
        if delta > 0.3:
            return "fortement surexposée"
        if delta > 0.1:
            return "légèrement surexposée"
        if delta < -0.3:
            return "fortement sous-exposée"
        if delta < -0.1:
            return "légèrement sous-exposée"
        return "correctement exposée"

    @staticmethod
    def _describe_reflectivity(value: float) -> str:
        if value > 0.7:
            return "très réfléchissante (type miroir)"
        if value > 0.4:
            return "modérément réfléchissante"
        return "mate ou faiblement réfléchissante"

    @staticmethod
    def _describe_material(mat: str) -> str:
        mapping = {
            "metal_speculaire": "Surface métallique à forte spécularité",
            "verre_texture": "Surface vitreuse texturée",
            "fourrure_textile": "Matière textile ou fourrure",
            "plastique_mat": "Plastique mat à faible brillance",
            "surface_generique": "Surface générique",
        }
        return mapping.get(mat, "Surface non classifiée")

    @staticmethod
    def _describe_gradient_error(err: float) -> str:
        if err > 0.4:
            return "Erreur de gradient élevée — correction agressive recommandée"
        if err > 0.2:
            return "Erreur de gradient modérée"
        return "Gradient cohérent"

    # ------------------------------------------------------------------
    # Main encoder
    # ------------------------------------------------------------------

    def encode(
        self,
        signature: dict,
        delta_metrics: dict | None = None,
        similar_corrections: list[dict] | None = None,
        relations: list | None = None,
        error_history: list[dict] | None = None,
    ) -> tuple[str, float]:
        """Build a semantic prompt and confidence score.

        Parameters
        ----------
        signature           : 8D object signature.
        delta_metrics       : Optional dict with gradient_error, ssim, brightness_delta, etc.
        similar_corrections : Previous corrections found by VisualMemory.
        relations           : Edges from SceneGraph for this object.
        error_history       : Previous errors to detect recurrence.

        Returns
        -------
        (prompt_str, confidence_float)
        """
        delta_metrics = delta_metrics or {}
        similar_corrections = similar_corrections or []
        relations = relations or []
        error_history = error_history or []

        lines: list[str] = []

        # Material
        mat = signature.get("material_class", "surface_generique")
        lines.append(f"Matériau détecté : {self._describe_material(mat)}.")

        # Reflectivity
        refl = signature.get("specularity", 0.0)
        lines.append(f"Réflectivité : {self._describe_reflectivity(refl)}.")

        # Brightness delta
        bd = delta_metrics.get("brightness_delta", 0.0)
        lines.append(f"Exposition : {self._describe_brightness(bd)} par rapport aux voisins.")

        # Gradient / SSIM
        ge = delta_metrics.get("gradient_error", 0.0)
        lines.append(self._describe_gradient_error(ge))

        ssim_val = delta_metrics.get("ssim")
        if ssim_val is not None:
            if ssim_val < 0.7:
                lines.append("Similarité structurelle faible — artefacts probables.")
            else:
                lines.append("Similarité structurelle acceptable.")

        # Previous corrections (with validated scene_hash)
        if similar_corrections:
            n = len(similar_corrections)
            lines.append(
                f"{n} correction(s) similaire(s) trouvée(s) en mémoire (scene_hash validé)."
            )

        # Scene relations (from SceneGraph light)
        if relations:
            rel_strs = [f"{r.source} {r.relation} {r.target}" for r in relations]
            lines.append(f"Relations scène : {', '.join(rel_strs)}.")

        # Confidence
        confidence = self._HIGH_CONF
        if ge > 0.4:
            confidence = self._MED_CONF
        repeated = len(error_history) >= 2
        if repeated:
            confidence = self._LOW_CONF
            lines.append("AVERTISSEMENT : même erreur répétée 2x — forcer modèle cloud.")

        lines.append(
            f"Confiance système : {'haute' if confidence > 0.7 else 'moyenne' if confidence > 0.4 else 'basse'}."
        )

        prompt = "\n".join(lines)

        # AGENT : format JSON forcé en fin de prompt
        prompt += (
            "\n\nRéponds UNIQUEMENT au format JSON :\n"
            '{"param": "nom_parametre", "delta": 0.0, "raison": "explication courte"}'
        )

        return prompt, confidence
