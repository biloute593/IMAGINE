"""Core greeting functions."""


def salut(name: str | None = None) -> str:
    """Return a friendly 'Salut' greeting message.

    Args:
        name: Optional name to greet. If None, returns a generic greeting.

    Returns:
        A greeting string.
    """
    if name:
        return f"Salut, {name} !"
    return "Salut !"


def greet(name: str | None = None, lang: str = "fr") -> str:
    """Return a greeting message in the specified language.

    Args:
        name: Optional name to greet.
        lang: Language code ('fr' for French, 'en' for English).

    Returns:
        A greeting string in the requested language.

    Raises:
        ValueError: If the language code is not supported.
    """
    greetings = {
        "fr": "Salut",
        "en": "Hello",
    }
    if lang not in greetings:
        supported = ", ".join(sorted(greetings))
        raise ValueError(
            f"Unsupported language '{lang}'. Supported: {supported}"
        )
    greeting = greetings[lang]
    if name:
        return f"{greeting}, {name} !"
    return f"{greeting} !"
