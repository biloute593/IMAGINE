"""
test_router.py — Tests unitaires pour HybridRouter (sans appels réseau réels)
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from photo_editor.hybrid_router import HybridRouter, CONFIDENCE_THRESHOLD


class TestParseJson:
    def test_valid_json(self):
        router = HybridRouter()
        result = router._parse_json('{"param": "color_delta", "delta": 0.3, "raison": "test"}')
        assert result["param"] == "color_delta"
        assert result["delta"] == pytest.approx(0.3)
        assert result["raison"] == "test"

    def test_json_with_surrounding_text(self):
        router = HybridRouter()
        text = 'Voici la réponse : {"param": "lumiere_ambiante", "delta": -0.1, "raison": "trop sombre"} fin.'
        result = router._parse_json(text)
        assert result["param"] == "lumiere_ambiante"
        assert result["delta"] == pytest.approx(-0.1)

    def test_invalid_json_returns_default(self):
        router = HybridRouter()
        result = router._parse_json("pas de json ici")
        assert result["param"] == "color_delta"
        assert result["delta"] == pytest.approx(0.0)

    def test_missing_fields_use_defaults(self):
        router = HybridRouter()
        result = router._parse_json('{"param": "texture_delta"}')
        assert result["param"] == "texture_delta"
        assert result["delta"] == pytest.approx(0.0)
        assert isinstance(result["raison"], str)


class TestShouldUseCloud:
    def test_low_confidence_triggers_cloud(self):
        router = HybridRouter()
        router._ollama_available = True
        assert router._should_use_cloud(0.3, "surface_generique") is True

    def test_high_confidence_no_cloud(self):
        router = HybridRouter()
        router._ollama_available = True
        assert router._should_use_cloud(0.9, "surface_generique") is False

    def test_complex_material_triggers_cloud(self):
        router = HybridRouter()
        router._ollama_available = True
        assert router._should_use_cloud(0.9, "fourrure_textile") is True
        assert router._should_use_cloud(0.9, "verre_texture") is True

    def test_ollama_unavailable_triggers_cloud(self):
        router = HybridRouter()
        router._ollama_available = False
        assert router._should_use_cloud(0.9, "plastique_mat") is True

    def test_confidence_at_threshold(self):
        router = HybridRouter()
        router._ollama_available = True
        # Exactement au seuil → cloud (< 0.7 est la condition)
        assert router._should_use_cloud(CONFIDENCE_THRESHOLD, "surface_generique") is False
        assert router._should_use_cloud(CONFIDENCE_THRESHOLD - 0.01, "surface_generique") is True


class TestRoute:
    def test_route_fallback_when_no_model(self, monkeypatch):
        """Vérifie que route() retourne une correction valide même sans modèle."""
        router = HybridRouter()
        router._ollama_available = False
        router._gemini_key = ""
        router._groq_key = ""

        result = router.route(
            "Test prompt", confidence=0.5, material_class="surface_generique"
        )
        assert "param" in result
        assert "delta" in result
        assert isinstance(result["delta"], float)
        assert "_model" in result

    def test_route_returns_model_key(self, monkeypatch):
        """Vérifie que le modèle utilisé est tracé dans la réponse."""
        router = HybridRouter()
        router._ollama_available = False
        router._gemini_key = ""
        router._groq_key = ""

        result = router.route("Test", confidence=0.9, material_class="plastique_mat")
        assert result["_model"] == "fallback"
