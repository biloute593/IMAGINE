"""
HybridRouter — Routing décisionnel entre modèle local (Qwen 14B Q4) et API cloud.

Modèle local  : Qwen 14B Q4 via Ollama (~10GB RAM)
Modèle cloud  : Gemini Free (GEMINI_API_KEY) ou Groq (GROQ_API_KEY)

AGENT : Ne jamais dégrader Qwen 14B → 7B pour "économiser de la RAM".
"""

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()


class HybridRouter:
    """Route LLM requests to local Ollama or cloud API based on confidence."""

    LOCAL_MODEL = "qwen2.5:14b"
    COMPLEX_MATERIALS = {"fourrure_textile", "verre_texture"}

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.groq_key = os.getenv("GROQ_API_KEY", "")

    # ------------------------------------------------------------------
    # Routing decision
    # ------------------------------------------------------------------

    def should_use_cloud(
        self,
        confidence: float,
        material_class: str = "",
        error_count: int = 0,
    ) -> bool:
        """Decide whether to route to cloud API.

        Routes to API if:
        - confidence < 0.7
        - material_class is complex (fourrure_textile, verre_texture)
        - error repeated (same problem 2x)
        - Ollama not available (fallback)
        """
        if confidence < 0.7:
            return True
        if material_class in self.COMPLEX_MATERIALS:
            return True
        if error_count >= 2:
            return True
        return False

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Robust JSON extraction between first { and last }.

        AGENT : Le LLM peut ajouter du texte autour du JSON.
        """
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    # ------------------------------------------------------------------
    # Query local model (Ollama)
    # ------------------------------------------------------------------

    def query_local(self, prompt: str) -> dict | None:
        """Query Qwen 14B Q4 via Ollama.

        Returns parsed JSON correction dict or None on failure.
        """
        try:
            import ollama
            response = ollama.chat(
                model=self.LOCAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response["message"]["content"]
            return self._parse_json(text)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Query cloud model (Gemini or Groq)
    # ------------------------------------------------------------------

    def query_cloud(self, prompt: str) -> dict | None:
        """Query Gemini or Groq cloud API.

        Falls back to Groq if Gemini is unavailable.
        """
        result = None
        if self.gemini_key:
            result = self._query_gemini(prompt)
        if result is None and self.groq_key:
            result = self._query_groq(prompt)
        return result

    def _query_gemini(self, prompt: str) -> dict | None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            return self._parse_json(response.text)
        except Exception:
            return None

    def _query_groq(self, prompt: str) -> dict | None:
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_key)
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            return self._parse_json(text)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Unified route
    # ------------------------------------------------------------------

    def route(
        self,
        prompt: str,
        confidence: float,
        material_class: str = "",
        error_count: int = 0,
    ) -> tuple[dict | None, str]:
        """Route prompt to the best available model.

        Returns
        -------
        (correction_dict | None, model_name)
        """
        use_cloud = self.should_use_cloud(confidence, material_class, error_count)

        if not use_cloud:
            result = self.query_local(prompt)
            if result is not None:
                return result, "qwen_local"

        # Fallback / cloud route
        result = self.query_cloud(prompt)
        if result is not None:
            return result, "cloud"

        # Ultimate fallback to local
        result = self.query_local(prompt)
        if result is not None:
            return result, "qwen_local"

        return None, "none"
