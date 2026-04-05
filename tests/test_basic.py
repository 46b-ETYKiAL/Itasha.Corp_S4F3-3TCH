"""Smoke tests for s4f3-etch package."""

from __future__ import annotations


def test_pydantic_importable():
    """Verify pydantic dependency is available."""
    import pydantic

    assert pydantic.VERSION is not None


def test_jinja2_importable():
    """Verify jinja2 dependency is available."""
    import jinja2

    env = jinja2.Environment(autoescape=True)
    template = env.from_string("Hello {{ name }}")
    assert template.render(name="world") == "Hello world"


def test_jinja2_autoescaping():
    """Verify Jinja2 autoescape prevents XSS."""
    import jinja2

    env = jinja2.Environment(autoescape=True)
    template = env.from_string("{{ user_input }}")
    result = template.render(user_input="<script>alert('xss')</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
