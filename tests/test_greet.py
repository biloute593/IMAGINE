"""Tests for the salut greeting module."""

import pytest

from salut import salut, greet


class TestSalut:
    def test_salut_without_name(self):
        assert salut() == "Salut !"

    def test_salut_with_name(self):
        assert salut("Alice") == "Salut, Alice !"


class TestGreet:
    def test_greet_french_default(self):
        assert greet() == "Salut !"

    def test_greet_french_with_name(self):
        assert greet("Bob", lang="fr") == "Salut, Bob !"

    def test_greet_english(self):
        assert greet(lang="en") == "Hello !"

    def test_greet_english_with_name(self):
        assert greet("Charlie", lang="en") == "Hello, Charlie !"

    def test_greet_unsupported_language(self):
        with pytest.raises(ValueError, match="Unsupported language 'de'"):
            greet(lang="de")
