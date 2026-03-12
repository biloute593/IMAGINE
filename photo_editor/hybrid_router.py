"""
hybrid_router.py — HybridRouter Qwen/Gemini/Groq
Route les requêtes vers le modèle local (Qwen 14B Q4 via Ollama) ou vers
une API cloud (Gemini / Groq) selon la confiance et le matériau.

RÈGLE 5 — ROUTING HYBRIDE :
  Qwen 14B Q4 par défaut. Ne pas dégrader à 7B.
  Gemini/Groq si confidence < 0.7 ou matériau complexe ou Ollama absent.

  AGENT : Ne jamais dégrader Qwen 14B → 7B pour "économiser de la RAM".
          14B Q4 = ~10GB. La machine a 16GB. C'est le bon choix.
"""

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Matériaux nécessitant un modèle cloud
COMPLEX_MATERIALS = {"fourrure_textile", "verre_texture"}

# Seuil de confiance pour basculer sur le cloud
CONFIDENCE_THRESHOLD = 0.7

# Modèle local principal
LOCAL_MODEL = "qwen2.5:14b"


class HybridRouter:
    """
    Dirige chaque requête de correction vers le modèle optimal.

    Routing vers API cloud si :
    - confidence < 0.7
    - material_class IN [fourrure_textile, verre_texture]
    - erreur récurrente (même problème 2x)
    - Ollama non disponible (fallback automatique)
    """

    def __init__(self) -> None:
        self._ollama_available: Optional[bool] = None
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")
        self._groq_key = os.getenv("GROQ_API_KEY", "")

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def route(
        self,
        prompt: str,
        confidence: float,
        material_class: str = "surface_generique",
    ) -> dict:
        """
        Envoie le prompt au modèle approprié et retourne la correction parsée.

        Args:
            prompt         : texte généré par SemanticEncoder.
            confidence     : score de confiance [0, 1].
            material_class : matériau de l'objet.

        Returns:
            dict {"param": str, "delta": float, "raison": str}
        """
        use_cloud = self._should_use_cloud(confidence, material_class)

        if use_cloud:
            model_name, response = self._call_cloud(prompt)
        else:
            model_name, response = self._call_local(prompt)

        correction = self._parse_json(response)
        correction["_model"] = model_name
        return correction

    # ------------------------------------------------------------------
    # Décision de routage
    # ------------------------------------------------------------------

    def _should_use_cloud(self, confidence: float, material_class: str) -> bool:
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info("Routing cloud : confidence=%.2f < %.2f", confidence, CONFIDENCE_THRESHOLD)
            return True
        if material_class in COMPLEX_MATERIALS:
            logger.info("Routing cloud : matériau complexe '%s'", material_class)
            return True
        if not self._is_ollama_available():
            logger.info("Routing cloud : Ollama indisponible")
            return True
        return False

    def _is_ollama_available(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            import ollama  # type: ignore
            ollama.list()
            self._ollama_available = True
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    # ------------------------------------------------------------------
    # Modèle local (Ollama / Qwen 14B Q4)
    # ------------------------------------------------------------------

    def _call_local(self, prompt: str) -> tuple:
        """
        Appelle Qwen 14B Q4 via Ollama.

        AGENT : Ne jamais dégrader à 7B.
        """
        try:
            import ollama  # type: ignore
            response = ollama.chat(
                model=LOCAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"]
            logger.debug("Ollama response: %s", content[:200])
            return LOCAL_MODEL, content
        except Exception as exc:
            logger.warning("Ollama error: %s — fallback cloud", exc)
            self._ollama_available = False
            return self._call_cloud(prompt)

    # ------------------------------------------------------------------
    # Modèle cloud (Gemini → Groq → fallback)
    # ------------------------------------------------------------------

    def _call_cloud(self, prompt: str) -> tuple:
        """Tente Gemini puis Groq, sinon retourne une correction neutre."""
        if self._gemini_key:
            try:
                return self._call_gemini(prompt)
            except Exception as exc:
                logger.warning("Gemini error: %s", exc)

        if self._groq_key:
            try:
                return self._call_groq(prompt)
            except Exception as exc:
                logger.warning("Groq error: %s", exc)

        logger.error("Aucun modèle disponible — correction neutre appliquée")
        return "fallback", '{"param": "color_delta", "delta": 0.0, "raison": "aucun modèle disponible"}'

    def _call_gemini(self, prompt: str) -> tuple:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=self._gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return "gemini-1.5-flash", response.text

    def _call_groq(self, prompt: str) -> tuple:
        from groq import Groq  # type: ignore
        client = Groq(api_key=self._groq_key)
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return "groq-llama-3.3-70b", chat.choices[0].message.content

    # ------------------------------------------------------------------
    # Parsing JSON robuste
    # ------------------------------------------------------------------

    def _parse_json(self, response: str) -> dict:
        """
        Extraction robuste du JSON entre { et } même si le LLM ajoute du texte.

        Format attendu : {"param": "...", "delta": 0.0, "raison": "..."}
        """
        # Cherche le premier bloc JSON valide
        match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return {
                    "param": str(data.get("param", "color_delta")),
                    "delta": float(data.get("delta", 0.0)),
                    "raison": str(data.get("raison", "")),
                }
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning("Impossible de parser la réponse LLM : %s", response[:200])
        return {"param": "color_delta", "delta": 0.0, "raison": "parsing failed"}
