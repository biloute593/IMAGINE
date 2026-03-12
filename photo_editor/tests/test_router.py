"""Tests for HybridRouter module."""

import json

import pytest

from photo_editor.hybrid_router import HybridRouter


class TestRoutingDecision:
    def test_low_confidence_uses_cloud(self):
        router = HybridRouter()
        assert router.should_use_cloud(confidence=0.5) is True

    def test_high_confidence_uses_local(self):
        router = HybridRouter()
        assert router.should_use_cloud(confidence=0.9) is False

    def test_complex_material_uses_cloud(self):
        router = HybridRouter()
        assert router.should_use_cloud(0.9, material_class="fourrure_textile") is True
        assert router.should_use_cloud(0.9, material_class="verre_texture") is True

    def test_simple_material_stays_local(self):
        router = HybridRouter()
        assert router.should_use_cloud(0.9, material_class="plastique_mat") is False

    def test_repeated_errors_use_cloud(self):
        router = HybridRouter()
        assert router.should_use_cloud(0.9, error_count=2) is True

    def test_boundary_confidence(self):
        router = HybridRouter()
        assert router.should_use_cloud(confidence=0.7) is False
        assert router.should_use_cloud(confidence=0.69) is True


class TestJsonParsing:
    def test_clean_json(self):
        text = '{"param": "color_delta", "delta": 0.1, "raison": "test"}'
        result = HybridRouter._parse_json(text)
        assert result is not None
        assert result["param"] == "color_delta"

    def test_json_with_surrounding_text(self):
        text = 'Here is my suggestion: {"param": "brightness", "delta": 0.2, "raison": "too dark"} Hope this helps!'
        result = HybridRouter._parse_json(text)
        assert result is not None
        assert result["param"] == "brightness"

    def test_invalid_json(self):
        text = "No JSON here at all"
        result = HybridRouter._parse_json(text)
        assert result is None

    def test_malformed_json(self):
        text = "{param: not valid json}"
        result = HybridRouter._parse_json(text)
        assert result is None


class TestRouteMethod:
    def test_route_returns_tuple(self):
        router = HybridRouter()
        # No LLM available, should return (None, "none")
        result, model = router.route("test prompt", confidence=0.9)
        assert model in ("qwen_local", "cloud", "none")
